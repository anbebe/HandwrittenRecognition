import tensorflow as tf

class ConvModel(tf.keras.Model):

  #1 Constructors
  def __init__(self, batch_norm=False, dropout=False, dropout_rate=0, L2_reg=0):
    super(ConvModel, self).__init__()
    #inherit functionality from parent class

    kernel_regularizer=tf.keras.regularizers.L2()

    #optimizer, loss function and metrics
    self.metrics_list = [
                        tf.keras.metrics.CategoricalCrossentropy(name="loss"),
                        tf.keras.metrics.CategoricalAccuracy(name="acc")
                       ]
    self.optimizer = tf.keras.optimizers.Adam()
    self.loss = tf.keras.losses.CategoricalCrossentropy()
    self.kernel_regularizer=tf.keras.regularizers.L2(L2_reg) if L2_reg else None
    self.dropout = dropout
    if self.dropout:
      self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)

    # layers to be used
    self.cnn_layers = []
    self.cnn_layers.append(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    self.cnn_layers.append(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    if batch_norm: 
      self.cnn_layers.append(tf.keras.layers.BatchNormalization())
    self.cnn_layers.append(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))
    self.cnn_layers.append(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    self.cnn_layers.append(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    if batch_norm: 
      self.cnn_layers.append(tf.keras.layers.BatchNormalization())
    self.cnn_layers.append(tf.keras.layers.GlobalAveragePooling2D())
    self.cnn_layers.append(tf.keras.layers.Dense(1024, activation='relu'))
    if batch_norm: 
      self.cnn_layers.append(tf.keras.layers.BatchNormalization())
    self.cnn_layers.append(tf.keras.layers.Dense(6, activation='softmax'))


  #2. call method (forward computation)
  @tf.function
  def call(self, x, training=False):
    for index, layer in enumerate(self.cnn_layers):
        x = layer(x)
        if self.dropout and index < len(self.cnn_layers)-1:
          x = self.dropout_layer(x, training)
    return x

  #3. metrics property
  @property
  def metrics(self):
    # return a list with all metrics in the model
    return self.metrics_list


  #4 reset all metrics object
  def reset_metrics(self):
    for metric in self.metrics:
      metric.reset_states()

  #5 training step method
  def train_step(self, data):
    # update the state of the metrics according to loss
    # return a dictionary with metrics name as keys an metric results
    img, label = data
    with tf.GradientTape() as tape:
      output = self(img, training=True)
      loss = self.loss(label, output)

    gradients = tape.gradient(loss, self.trainable_variables)

    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    #update the state of the metrics according to loss
    for metric in self.metrics:
            metric.update_state(label, output)

    # return a dictionary with metric names as keys and metric results as values
    return {m.name : m.result() for m in self.metrics}
    

  #6. test step method
  def test_step(self, data):
    img, label = data
    output = self(img, training=False)
    loss = self.loss(label, output)
    for metric in self.metrics:
      metric.update_state(label, output)

    return {"val_"+m.name : m.result() for m in self.metrics}