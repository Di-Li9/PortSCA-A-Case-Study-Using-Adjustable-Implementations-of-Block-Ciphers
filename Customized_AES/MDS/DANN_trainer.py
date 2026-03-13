import sys
import os
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import Progbar
from tensorflow.keras.metrics import Mean
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.losses import categorical_crossentropy

# Changed import names
from DANN_Model import build_feature_extractor
from DANN_Model import build_label_classify_extractor  
from DANN_Model import build_domain_classify_extractor
from GradientReveresalLayer import GradientReversalLayer
from utils.model_utils import grl_lambda_schedule
from utils.model_utils import learning_rate_schedule

class DANN(object):

    def __init__(self,config):
        """
        Initialization function for Domain Adaptation Neural Network (DANN) between MNIST and MNIST_M
        :param config: Configuration parameters class
        """
        # Initialize configuration parameters
        self.cfg = config
        # Define placeholder for GRL lambda
        self.grl_lambd = 1.0            # Gradient Reversal Layer parameter
        # Build DANN architecture
        self.build_DANN()
        # Define loss function and accuracy metrics
        self.loss = categorical_crossentropy
        self.acc = categorical_accuracy

        # Initialize metrics for training and validation tracking
        self.train_loss = Mean("train_loss", dtype=tf.float32)
        self.train_trace_cls_loss = Mean("train_trace_cls_loss", dtype=tf.float32)
        self.train_domain_cls_loss = Mean("train_domain_cls_loss", dtype=tf.float32)
        self.train_trace_cls_acc = Mean("train_trace_cls_acc", dtype=tf.float32)
        self.train_domain_cls_acc = Mean("train_domain_cls_acc", dtype=tf.float32)
        self.val_loss = Mean("val_loss", dtype=tf.float32)
        self.val_trace_cls_loss = Mean("val_trace_cls_loss", dtype=tf.float32)
        self.val_domain_cls_loss = Mean("val_domain_cls_loss", dtype=tf.float32)
        self.val_trace_cls_acc = Mean("val_trace_cls_acc", dtype=tf.float32)
        self.val_domain_cls_acc = Mean("val_domain_cls_acc", dtype=tf.float32)

        # Define optimizer with initial learning rate
        self.optimizer = tf.keras.optimizers.Adam(self.cfg.init_learning_rate)

    def build_DANN(self):
        """
        Construct the Domain-Adversarial Neural Network architecture
        """
        # Define input layer
        self.trace_input = Input(shape=self.cfg.input_shape,name="trace_input")

        # Shared feature extractor between domain and class classifiers
        self.feature_encoder = build_feature_extractor()
        # Task-specific classifiers
        self.trace_cls_encoder = build_label_classify_extractor()
        self.domain_cls_encoder = build_domain_classify_extractor()

        # Gradient Reversal Layer for domain adaptation
        self.grl = GradientReversalLayer()

        # Build end-to-end DANN model
        self.dann_model = Model(self.trace_input,
                                [self.trace_cls_encoder(self.feature_encoder(self.trace_input)),
                                 self.domain_cls_encoder(self.grl(self.feature_encoder(self.trace_input)))])
        
        # Load pre-trained weights if specified
        if self.cfg.pre_model_path is not None:
            print('Loading pre-trained weights...')
            # Temporarily disable logging
            tf.get_logger().setLevel('ERROR')
            self.dann_model.load_weights(self.cfg.pre_model_path,by_name=True,skip_mismatch=True)

    def train(self,train_source_datagen,train_target_datagen,
              val_target_datagen,train_iter_num,val_iter_num):
        """
        Main training loop for DANN
        :param train_source_datagen: Source domain training data generator
        :param train_target_datagen: Target domain training data generator
        :param val_target_datagen: Validation data generator (used for evaluation only)
        :param train_iter_num: Number of iterations per training epoch
        :param val_iter_num: Number of iterations per validation run
        """
        # Set up checkpoint directory
        checkpoint_dir = self.cfg.checkpoints_dir
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        
        print('\n----------- Starting training -----------\n')
        best_val_loss = float('inf')  # Initialize with infinity

        for ep in np.arange(1, self.cfg.epoch+1, 1):
            # Initialize progress bar
            self.progbar = Progbar(train_iter_num+1)
            print(f'Epoch {ep}/{self.cfg.epoch}')

            # Train for one epoch
            train_loss, train_trace_cls_acc = self.train_one_epoch(
                train_source_datagen, 
                train_target_datagen,
                train_iter_num,
                ep)
            
            # Validate using separate validation set (no gradient updates)
            val_loss, val_trace_cls_acc = self.eval_one_epoch(
                val_target_datagen,
                val_iter_num,
                ep)

            # Update progress bar with validation metrics
            self.progbar.update(train_iter_num+1, [
                ('target_loss', val_loss),
                ("target_acc", val_trace_cls_acc)
            ])

            # Save model only if validation performance improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_name = f"Transfer_Model(Original MDS).h5"
                self.dann_model.save(os.path.join(checkpoint_dir, model_name))

            # Reset metrics for next epoch
            self.train_loss.reset_states()
            self.train_trace_cls_acc.reset_states()
            self.train_domain_cls_loss.reset_states()
            self.train_domain_cls_acc.reset_states()
            self.val_loss.reset_states()
            self.val_trace_cls_acc.reset_states()
            self.val_domain_cls_loss.reset_states()
            self.val_domain_cls_acc.reset_states()

            # Log epoch statistics
            log_str = (
                f"Epoch{ep:03d}-train_loss-{train_loss:.3f}-val_loss-{val_loss:.3f}-"
                f"train_trace_cls_acc-{train_trace_cls_acc:.3f}-val_trace_cls_acc-{val_trace_cls_acc:.3f}"
            )
            print(log_str)
              
        print('\n----------- Training completed -----------\n')

    def train_one_epoch(self,train_source_datagen,train_target_datagen,train_iter_num,ep):
        """
        Single epoch training procedure
        :param train_source_datagen: Source domain training data generator
        :param train_target_datagen: Target domain training data generator
        :param train_iter_num: Number of training iterations per epoch
        :param ep: Current epoch number
        """
        for i in np.arange(1, train_iter_num + 1):
            # Get mini-batches from both domains
            batch_mnist_trace_data, batch_mnist_labels = train_source_datagen.__next__()
            batch_mnist_m_trace_data, batch_mnist_m_labels = train_target_datagen.__next__()

            # Create domain labels (source=0, target=1)
            batch_domain_labels = np.vstack([np.tile([1., 0.], [len(batch_mnist_labels), 1]),
                                             np.tile([0., 1.], [len(batch_mnist_m_labels), 1])]).astype(np.float32)
            batch_trace_data = np.concatenate([batch_mnist_trace_data, batch_mnist_m_trace_data], axis=0)

            # Update adaptive parameters
            iter = (ep - 1) * train_iter_num + i
            process = iter * 1.0 / (self.cfg.epoch * train_iter_num)
            self.grl_lambd = grl_lambda_schedule(process)  # Update GRL lambda
            learning_rate = learning_rate_schedule(process, init_learning_rate=self.cfg.init_learning_rate)
            tf.keras.backend.set_value(self.optimizer.lr, learning_rate)

            # Training step with gradient updates
            with tf.GradientTape() as tape:
                # Forward pass for source domain classification
                trace_cls_feature = self.feature_encoder(batch_mnist_trace_data)
                trace_cls_pred = self.trace_cls_encoder(trace_cls_feature,training=True)
                trace_cls_loss = self.loss(batch_mnist_labels,trace_cls_pred)
                trace_cls_acc = self.acc(batch_mnist_labels, trace_cls_pred)

                # Forward pass for domain classification
                domain_cls_feature = self.feature_encoder(batch_trace_data)
                domain_cls_pred = self.domain_cls_encoder(self.grl(domain_cls_feature, self.grl_lambd),
                                                                  training=True)
                domain_cls_loss = self.loss(batch_domain_labels, domain_cls_pred)
                domain_cls_acc = self.acc(batch_domain_labels, domain_cls_pred)

                # Total loss calculation
                loss = tf.reduce_mean(trace_cls_loss) + tf.reduce_mean(domain_cls_loss)

            # Backpropagation and parameter updates
            vars = tape.watched_variables()
            grads = tape.gradient(loss, vars)
            self.optimizer.apply_gradients(zip(grads, vars))

            # Update metrics
            self.train_loss(loss)
            self.train_trace_cls_loss(trace_cls_loss)
            self.train_domain_cls_loss(domain_cls_loss)
            self.train_trace_cls_acc(trace_cls_acc)
            self.train_domain_cls_acc(domain_cls_acc)

            # Update progress bar
            self.progbar.update(i, [('loss', loss),
                               ('source_train_loss', trace_cls_loss),
                               ('domain_train_loss', domain_cls_loss),
                               ("source_train_acc", trace_cls_acc),
                               ("domain_train_acc", domain_cls_acc)])
        
        return self.train_loss.result(),self.train_trace_cls_acc.result()

    def eval_one_epoch(self,val_target_datagen,val_iter_num,ep):
        """
        Validation procedure (no parameter updates)
        :param val_target_datagen: Validation data generator
        :param val_iter_num: Number of validation iterations
        :param ep: Current epoch number
        """
        for i in np.arange(1, val_iter_num + 1):
            # Get validation batch
            batch_mnist_m_trace_data, batch_mnist_m_labels = val_target_datagen.__next__()
            batch_mnist_m_domain_labels = np.tile([0., 1.], [len(batch_mnist_m_labels), 1]).astype(np.float32)

            # Forward pass without gradient computation
            # Note: training=False for batch norm/dropout layers
            target_trace_feature = self.feature_encoder(batch_mnist_m_trace_data)
            target_trace_cls_pred = self.trace_cls_encoder(target_trace_feature, training=False)
            target_domain_cls_pred = self.domain_cls_encoder(target_trace_feature, training=False)

            # Loss calculations (no backpropagation)
            target_trace_cls_loss = self.loss(batch_mnist_m_labels,target_trace_cls_pred)
            target_domain_cls_loss = self.loss(batch_mnist_m_domain_labels,target_domain_cls_pred)
            target_loss = tf.reduce_mean(target_trace_cls_loss) + tf.reduce_mean(target_domain_cls_loss)
            
            # Accuracy calculations
            trace_cls_acc = self.acc(batch_mnist_m_labels, target_trace_cls_pred)
            domain_cls_acc = self.acc(batch_mnist_m_domain_labels, target_domain_cls_pred)

            # Update validation metrics
            self.val_loss(target_loss)
            self.val_trace_cls_loss(target_trace_cls_loss)
            self.val_domain_cls_loss(domain_cls_acc)
            self.val_trace_cls_acc(trace_cls_acc)
            self.val_domain_cls_acc(domain_cls_acc)

        return self.val_loss.result(), self.val_trace_cls_acc.result()