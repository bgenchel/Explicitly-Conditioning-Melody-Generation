import json
import os
import os.path as op
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

from . import constants as const
from .data.datasets import BebopPitchDurDataset
from .data.dataloaders import SplitDataLoader
from .logging import TensorBoardWriter


class Trainer:
    def __init__(self, model, args):
        """
        :param args: an argparse object from the argparse class
        """
        self.model = model
        self.args = args

        self.info_dict = OrderedDict()
        self.info_dict['run_datetime'] = self.args.run_datetime_str
        self.info_dict.update(vars(self.args))

        dataset = BebopPitchDurDataset(seq_len=self.args.seq_len, train_type=self.args.train_type)
        self.train_loader, self.valid_loader = SplitDataLoader(dataset, batch_size=self.args.batch_size, 
                                                               drop_last=True).split()

        params = self.model.parameters()
        self.optimizer = optim.Adam(params, lr=self.args.learning_rate, amsgrad=True)
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.9, patience=3, verbose=True)
        self.loss_fn = nn.NLLLoss()

        self.dirpath = op.join(os.getcwd(), "runs", args.model)
        if self.args.keep:
            self.dirpath = op.join(self.dirpath, self.args.title)
        else:
            self.dirpath = op.join(self.dirpath, "test_runs", self.args.title)
        
    def train_model(self, print_interval=250):
        """
        :@param: train_type - can be either 'next_step' of 'full_sequence'
        """
        writer = TensorBoardWriter(op.join(self.dirpath, 'tensorboard'))
        self.save_model_inputs()

        min_valid_loss = float('inf')
        best_model = (None, None, None)
        train_losses, valid_losses = [], []

        print("Beginning Training - %s model" % self.args.model)
        print("Cuda available: ", torch.cuda.is_available())
        try:
            # write the initial losses
            init_train_loss = self.eval_epoch(self.train_loader)  
            train_losses.append(init_train_loss)
            print("Initial Training Loss: %.5f" % (init_train_loss))

            init_valid_loss = self.eval_epoch(self.valid_loader)
            valid_losses.append(init_valid_loss)
            print("Initial Validation Loss: %.5f" % (init_valid_loss))

            writer.write_loss({'training': init_train_loss, 'validation': init_valid_loss}, 0)
            # train
            for epoch in range(self.args.epochs):
                epoch_label = epoch + 1
                print("="*20 + "\nEpoch %d / %d\n" % (epoch_label, self.args.epochs) + "="*20)

                train_loss = self.train_epoch(self.train_loader, print_interval)
                train_losses.append(train_loss)
                print("Epoch %d Training Loss: %.5f" % (epoch_label, train_loss))

                valid_loss = self.eval_epoch(self.valid_loader)
                valid_losses.append(valid_loss)
                print("Epoch %d Validation Loss: %.5f" % (epoch_label, valid_loss))

                if valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    best_model = (epoch_label, train_loss, valid_loss)
                    self.save_model()

                # self.scheduler.step(valid_loss)
                writer.write_loss({'training': train_loss, 'validation': valid_loss}, epoch + 1)
            print("Finished Training.")
        except KeyboardInterrupt:
            print("Training Interrupted.")

        writer.close()
        print("Best Model - [ Epoch %d, Training Loss: %.5f, Validation Loss: %.5f ]" % best_model)
        self.save_run(best_model, train_losses, valid_losses)

    def train_epoch(self, data_iter, print_interval):
        batch_count = 0
        interval_loss = 0.0
        epoch_loss = 0.0
        for data_batch, target_batch in data_iter:
            # get the data in the right form
            data_batch = self.model.data_assembler(data_batch)
            target_batch = self.model.target_assembler(target_batch)
            # detach hidden state
            self.model.init_hidden_and_cell(self.args.batch_size)
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # forward pass
            if self.args.train_type == "next_step":
                output = self.model(data_batch)[:, -1, :]
            elif self.args.train_type == "full_sequence":
                output = self.model(data_batch)
            # backward + optimize
            loss = self.loss_fn(output, target_batch)
            loss.backward()
            self.optimizer.step()
            # print stats out
            epoch_loss += float(loss.item())
            interval_loss += float(loss.item())
            if batch_count % print_interval == print_interval - 1:
                print("Batch: %d, Loss: %.5f" % (batch_count + 1, interval_loss / print_interval))
                interval_loss = 0.0
            batch_count += 1
        avg_epoch_loss = epoch_loss / batch_count
        return avg_epoch_loss

    def eval_epoch(self, data_iter):
        total_loss = 0.0
        batch_count = 0
        with torch.no_grad():
            for data_batch, target_batch in data_iter:
                data_batch = self.model.data_assembler(data_batch)
                target_batch = self.model.target_assembler(target_batch)
                self.model.init_hidden_and_cell(self.args.batch_size)
                output = self.model(data_batch)[:, -1, :]
                loss = self.loss_fn(output, target_batch)
                total_loss += float(loss.item())
                batch_count += 1
        avg_loss = total_loss / batch_count
        return avg_loss

    def save_model_inputs(self):
        if self.args.keep:
            if not op.exists(self.dirpath):
                os.makedirs(self.dirpath)

            print('Saving model inputs ...')
            if not op.exists(op.join(self.dirpath, 'model_inputs.json')):
                model_inputs = {'hidden_dim': self.args.hidden_dim,
                                'seq_len': self.args.seq_len,
                                'batch_size': self.args.batch_size,
                                'dropout': self.args.dropout,
                                'batch_norm': self.args.batch_norm,
                                'no_cuda': self.args.no_cuda}
                json.dump(model_inputs, open(op.join(self.dirpath, 'model_inputs.json'), 'w'), indent=4)

    def save_model(self):
        if self.args.keep:
            if not op.exists(self.dirpath):
                os.makedirs(self.dirpath)

            print('Saving model ...')
            torch.save(self.model.state_dict(), op.join(self.dirpath, 'model_state.pt'))

    def save_run(self, best_model, train_losses, valid_losses):
        if not op.exists(self.dirpath):
            os.makedirs(self.dirpath)
        
        self.info_dict['best_model'] = {'epoch': best_model[0], 'train_loss': best_model[1], 'valid_loss': best_model[2]}
        self.info_dict['epochs_completed'] = len(train_losses)
        self.info_dict['final_training_loss'] = train_losses[-1]
        self.info_dict['final_valid_loss'] = valid_losses[-1]

        print('Writing run info file ...')
        with open(op.join(self.dirpath, 'info.txt'), 'w') as fp:
            max_kw_len = max([len(key) for key in self.info_dict.keys()])
            for k, v in self.info_dict.items():
                space_buffer = ' '*(max_kw_len - len(k))
                fp.write('%s:%s\t %s\n'%(str(k), space_buffer, str(v)))
            fp.close()

        if self.args.keep:
            print('Writing training losses ...') 
            json.dump(train_losses, open(op.join(self.dirpath, 'train_losses.json'), 'w'), indent=4)

            print('Writing validation losses ...') 
            json.dump(valid_losses, open(op.join(self.dirpath, 'valid_losses.json'), 'w'), indent=4)
        return
