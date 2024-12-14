# Acknowledgements:
# This code includes implementations and adaptations inspired by the work and contributions of https://github.com/Dadaism6/CS260D-ADI.
from utils import Adahessian
import os
from .subset_trainer import *
from transformers import AdamW


class NLPCRESTTrainer(NLPSubsetTrainer):
    def __init__(
        self,
        args: argparse.Namespace,
        model: nn.Module,
        train_dataset: IndexedDataset,
        val_loader: DataLoader,
        train_weights: torch.Tensor = None,
    ):
        super().__init__(args, model, train_dataset, val_loader, train_weights)
        self.train_indices = np.arange(len(self.train_dataset))
        self.steps_per_epoch = np.ceil(int(len(self.train_dataset) * self.args.train_frac) / self.args.batch_size).astype(int)
        self.reset_step = self.steps_per_epoch
        print(f"reset step: {self.reset_step}")
        self.random_sets = np.array([])

        self.num_checking = 0

        self.gradient_approx_optimizer = Adahessian(self.model.parameters())

        self.loss_watch = np.ones((self.args.watch_interval, len(self.train_dataset))) * -1

        self.approx_time = AverageMeter()
        self.compare_time = AverageMeter()
        self.similarity_time = AverageMeter()
        self.delta = [torch.zeros_like(p.data) for p in self.model.parameters()]

    def _train_epoch(self, epoch: int):
        """
        Train the model for one epoch
        :param epoch: current epoch
        """
        self.model.train()
        self._reset_metrics()

        lr = self.lr_scheduler.get_last_lr()[0]
        self.args.logger.info(f"Epoch {epoch} LR {lr:.6f}")
        print(torch.cuda.memory_summary())

        for training_step in range(self.steps_per_epoch * epoch, self.steps_per_epoch * (epoch + 1)):

            if (training_step > self.reset_step) and ((training_step - self.reset_step) % self.args.check_interval == 0):
                self._check_approx_error(epoch, training_step)

            if training_step == self.reset_step:
                self._save_checkpoint(epoch=float(f"{epoch}.{training_step}"))
                self.used_indices = set()
                self._select_subset(epoch, training_step)
                self._update_train_loader_and_weights()
                self.train_iter = iter(self.train_loader)
                self._get_quadratic_approximation(epoch, training_step)
                
            elif training_step == 0:
                self.train_iter = iter(self.train_loader)

            if training_step % 50 == 0:
                torch.cuda.empty_cache()
                print(f"On epoch {epoch} step {training_step}")

            data_start = time.time()
            try:
                batch = next(self.train_iter)
            except StopIteration:
                if self.args.cache_dataset and self.args.clean_cache_iteration:
                    self.train_dataset.clean()
                    self._update_train_loader_and_weights()
                self.train_iter = iter(self.train_loader)
                batch = next(self.train_iter)

            data_time = time.time() - data_start
            self.batch_data_time.update(data_time)

            loss, train_acc = self._forward_and_backward(batch)

            data_start = time.time()

            if self.args.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "training_step": training_step,
                    "train_loss": loss.item(),
                    "train_acc": train_acc})

    def _forward_and_backward(self, batch):
        self.optimizer.zero_grad()
        input_ids = batch['input_ids'].to(self.args.device)
        attention_mask = batch['attention_mask'].to(self.args.device)
        labels = batch['label'].to(self.args.device)
        data_idx = batch['index']
        # train model with the current batch and record forward and backward time
        forward_start = time.time()
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        forward_time = time.time() - forward_start
        self.batch_forward_time.update(forward_time)

        loss = output.loss
        loss = (loss * self.train_weights[data_idx]).mean()

        lr = self.lr_scheduler.get_last_lr()[0]
        if lr > 0:
            # compute the parameter change delta
            self.model.zero_grad()
            # approximate with hessian diagonal
            loss.backward(create_graph=True)

            gf_current, _, _ = self.gradient_approx_optimizer.step(momentum=False)

            # Split gf_current into chunks based on the sizes of each parameter
            gf_chunks = gf_current.split([p.numel() for p in self.model.parameters()])
            for d, gf, p in zip(self.delta, gf_chunks, self.model.parameters()):
                d -= lr * gf.view_as(p)  # Reshape gf to match the parameter's shape

        backward_start = time.time()
        loss.backward()
        self.optimizer.step()
        backward_time = time.time() - backward_start
        self.batch_backward_time.update(backward_time)

        # update training loss and accuracy
        preds = output.logits.argmax(dim=-1)
        train_acc = (preds == labels).float().mean().item()
        self.train_loss.update(loss.item(), input_ids.size(0))
        self.train_acc.update(train_acc, input_ids.size(0))

        return loss, train_acc
    
    def _get_quadratic_approximation(self, epoch: int, training_step: int):
        """
        Compute the quadratic approximation of the loss function
        :param epoch: current epoch
        :param training_step: current training step
        """
        self.args.logger.info(f"geting quadratic approximation at epoch {epoch} step {training_step}")
        if self.args.approx_with_coreset:
            # Update the second-order approximation with the coreset
            approx_loader = DataLoader(
                Subset(self.train_dataset, indices=self.subset),
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                pin_memory=True,
                sampler=None,
            )
        else:
            # Update the second-order approximation with random subsets
            approx_loader = DataLoader(
                Subset(self.train_dataset, indices=self.random_sets),
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                pin_memory=True,
                sampler=None,
            )

        approx_start = time.time()
        curvature_norm = AverageMeter()
        self.start_loss = AverageMeter()

        for approx_batch, batch in enumerate(approx_loader):
            input_ids = batch['input_ids'].to(self.args.device)
            attention_mask = batch['attention_mask'].to(self.args.device)
            labels = batch['label'].to(self.args.device)
            idx = batch['index']
            # compute output
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            if self.args.approx_with_coreset:
                loss = output.loss
                loss = (loss * self.train_weights[idx]).mean()
            else:
                loss = output.loss
            self.model.zero_grad()

            # approximate with hessian diagonal
            loss.backward(create_graph=True)
            gf_tmp, ggf_tmp, ggf_tmp_moment = self.gradient_approx_optimizer.step(momentum=True)

            if approx_batch == 0:
                self.gf = gf_tmp * len(idx)
                self.ggf = ggf_tmp * len(idx)
                self.ggf_moment = ggf_tmp_moment * len(idx)
            else:
                self.gf += gf_tmp * len(idx)
                self.ggf += ggf_tmp * len(idx)
                self.ggf_moment += ggf_tmp_moment * len(idx)

            curvature_norm.update(ggf_tmp_moment.norm())
            self.start_loss.update(loss.item(), input_ids.size(0))

        approx_time = time.time() - approx_start
        self.approx_time.update(approx_time)

        self.gf /= len(approx_loader.dataset)
        self.ggf /= len(approx_loader.dataset)
        self.ggf_moment /= len(approx_loader.dataset)
        self.delta = [torch.zeros_like(p.data) for p in self.model.parameters()]

        gff_norm = curvature_norm.avg
        self.start_loss = self.start_loss.avg
        if self.args.approx_moment:
            self.ggf = self.ggf_moment

        if training_step == self.steps_per_epoch:
            self.init_curvature_norm = gff_norm
        else:
            self.args.check_interval = int(torch.ceil(self.init_curvature_norm / gff_norm * self.args.interval_mul))
            self.args.num_minibatch_coreset = min(self.args.check_interval * self.args.batch_num_mul, self.steps_per_epoch)
        self.args.logger.info(
            f"Checking interval {self.args.check_interval}. Number of minibatch coresets {self.args.num_minibatch_coreset}")
        if self.args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'training_step': training_step,
                'ggf_norm': gff_norm,
                'check_interval': self.args.check_interval,
                'num_minibatch_coreset': self.args.num_minibatch_coreset})

    def _check_approx_error(self, epoch: int, training_step: int) -> torch.Tensor:

        """

        Compute the quadratic approximation of the loss over a chosen subset.

        This step refines the second-order approximation used by CREST.

        """
        self.args.logger.info(f"checking approx error at epoch {epoch} step {training_step}")
        self.num_checking += 1

        # Timing variables
        start_total = time.time()

        # Step 1: Get train output
        start_train_output = time.time()
        self._get_train_output()
        train_output_time = time.time() - start_train_output
        self.args.logger.info(f"Time for _get_train_output: {train_output_time:.3f} seconds")

        # Step 2: Compute true_loss
        start_true_loss = time.time()
        true_loss = self.val_criterion(
            torch.from_numpy(self.train_output[self.random_sets]),
            torch.from_numpy(self.train_target[self.random_sets])
        )
        true_loss_time = time.time() - start_true_loss
        self.args.logger.info(f"Time for computing true_loss: {true_loss_time:.3f} seconds")

        # Step 3: Compute delta_norm
        start_delta_norm = time.time()
        delta_norm = torch.sqrt(sum(torch.norm(d)**2 for d in self.delta))
        delta_norm_time = time.time() - start_delta_norm
        self.args.logger.info(f"Time for computing delta_norm: {delta_norm_time:.3f} seconds")

        # Step 4: Prepare tensors for matmul
        start_tensor_prep = time.time()
        delta_tensor = torch.cat([d.flatten() for d in self.delta], dim=0)
        tensor_prep_time = time.time() - start_tensor_prep
        self.args.logger.info(f"Time for preparing tensors: {tensor_prep_time:.3f} seconds")

        # Step 5: Compute approx_loss
        start_approx_loss = time.time()
        approx_loss = torch.matmul(delta_tensor, self.gf) + self.start_loss
        approx_loss += 0.5 * torch.matmul(delta_tensor * self.ggf, delta_tensor)
        approx_loss_time = time.time() - start_approx_loss
        self.args.logger.info(f"Time for computing approx_loss: {approx_loss_time:.3f} seconds")

        # Step 6: Compute loss_diff
        start_loss_diff = time.time()
        loss_diff = abs(true_loss - approx_loss.item())
        loss_diff_time = time.time() - start_loss_diff
        self.args.logger.info(f"Time for computing loss_diff: {loss_diff_time:.3f} seconds")

        # Step 7: Log results
        thresh = self.args.check_thresh_factor * true_loss
        log_str = f"Iter {training_step} loss difference {loss_diff:.3f} threshold {thresh:.3f} True loss {true_loss:.3f} Approx loss {approx_loss.item():.3f} Delta norm {delta_norm:.3f}"
        if loss_diff > thresh:
            self.reset_step = training_step
            log_str += f" is larger than threshold {thresh:.3f}. "
        self.args.logger.info(log_str)

        # Step 8: Update compare time
        compare_time = time.time() - start_total
        self.compare_time.update(compare_time)
        self.args.logger.info(f"Total time for _check_approx_error: {compare_time:.3f} seconds")

        # Step 9: Log to WandB if applicable
        if self.args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'training_step': training_step,
                'loss_diff': loss_diff,
                'loss_thresh': thresh,
                'delta_norm': delta_norm,
                'num_checking': self.num_checking})


    def _drop_learned_data(self, epoch: int, training_step: int, indices: np.ndarray):
        """
        Drop the learned data points
        :param epoch: current epoch
        :param training_step: current training step
        :param indices: indices of the data points that have valid predictions
        """

        self.loss_watch[epoch % self.args.watch_interval, indices] = self.train_criterion(
            torch.from_numpy(self.train_output[indices]), torch.from_numpy(self.train_target[indices]).long()).numpy()

        if ((epoch + 1) % self.args.drop_interval == 0):
            order_ = np.where(np.sum(self.loss_watch > self.args.drop_thresh, axis=0) > 0)[0]
            unselected = np.where(np.sum(self.loss_watch >= 0, axis=0) == 0)[0]
            order_ = np.concatenate([order_, unselected])

            order = []
            per_class_size = int(np.ceil(self.args.random_subset_size * self.args.train_size / self.args.num_classes))
            for c in np.unique(self.train_target):
                class_indices_new = np.intersect1d(np.where(self.train_target == c)[0], order_)
                if len(class_indices_new) > per_class_size:
                    order.append(class_indices_new)
                else:
                    class_indices = np.intersect1d(np.where(self.train_target == c)[0], self.train_indices)
                    order.append(class_indices)
            order = np.concatenate(order)

            if len(order) > self.args.min_train_size:
                self.train_indices = order

            if self.args.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'forgettable_train': len(self.train_indices)})

    def _select_random_set(self) -> np.ndarray:
        indices = []
        for c in np.unique(self.train_target):
            class_indices = np.intersect1d(np.where(self.train_target == c)[0], self.train_indices)
            indices_per_class = np.random.choice(class_indices, size=int(
                np.ceil(self.args.random_subset_size * self.args.train_size / self.args.num_classes)), replace=False)
            indices.append(indices_per_class)
        indices = np.concatenate(indices)

        return indices

    def _select_subset(self, epoch: int, training_step: int):
        """
        Select a subset of the data
        """
        self.args.logger.info(f"selecting subset at epoch {epoch} step {training_step}")
        start_total = time.time()  # Start timing the total function execution

        # Step 1: Call parent method
        start_parent = time.time()
        super()._select_subset(epoch, training_step)
        parent_time = time.time() - start_parent
        self.args.logger.info(f"Time for parent _select_subset: {parent_time:.3f} seconds")

        # Step 2: Get random subsets
        start_random = time.time()
        self.random_sets = []
        self.subset = []
        self.subset_weights = []
        for _ in range(int(self.args.num_minibatch_coreset)):
            random_subset = self._select_random_set()
            self.random_sets.append(random_subset)

        random_time = time.time() - start_random
        self.args.logger.info(f"Time for selecting random subsets: {random_time:.3f} seconds")

        # Step 3: Create DataLoader
        start_dataloader = time.time()
        self.train_val_loader = DataLoader(
            Subset(self.train_dataset, indices=np.concatenate(self.random_sets)),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        dataloader_time = time.time() - start_dataloader
        self.args.logger.info(f"Time for creating DataLoader: {dataloader_time:.3f} seconds")

        # Step 4: Get train output
        start_train_output = time.time()
        self._get_train_output()
        train_output_time = time.time() - start_train_output
        self.args.logger.info(f"Time for _get_train_output: {train_output_time:.3f} seconds")

        # Step 5: Drop learned data points
        start_drop = time.time()
        if self.args.drop_learned:
            self._drop_learned_data(epoch, training_step, np.concatenate(self.random_sets))
        drop_time = time.time() - start_drop
        self.args.logger.info(f"Time for dropping learned data points: {drop_time:.3f} seconds")

        # Step 6: Process random sets and generate subsets
        start_generate = time.time()
        for random_set in self.random_sets:
            preds = self.train_softmax[random_set]
            if np.shape(preds)[-1] == self.args.num_classes:
                preds -= np.eye(self.args.num_classes)[self.train_target[random_set]]

            (
                subset,
                weight,
                _,
                similarity_time,
            ) = self.subset_generator.generate_subset(
                preds=preds,
                epoch=epoch,
                B=self.args.batch_size,
                idx=random_set,
                targets=self.train_target,
                use_submodlib=(self.args.smtk == 0),
            )
            self.similarity_time.update(similarity_time)

            self.subset.append(subset)
            self.subset_weights.append(weight)
        generate_time = time.time() - start_generate
        self.args.logger.info(f"Time for processing random sets and generating subsets: {generate_time:.3f} seconds")

        # Step 7: Final processing and saving
        start_final = time.time()
        self.subset = np.concatenate(self.subset)
        self.subset_weights = np.concatenate(self.subset_weights)
        self.random_sets = np.concatenate(self.random_sets)
        if self.args.save_subset:
            save_path = os.path.join(self.args.save_subset_dir, f"subset_epoch{epoch}_step{training_step}.npz")
            np.savez(save_path, subset=self.subset, subset_weights=self.subset_weights, random_sets=self.random_sets)
        final_time = time.time() - start_final
        self.args.logger.info(f"Time for final processing and saving: {final_time:.3f} seconds")

        # Total time
        total_time = time.time() - start_total
        self.args.logger.info(f"Total time for _select_subset: {total_time:.3f} seconds")

