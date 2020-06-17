import tensorflow as tf

loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def disc_loss(real_output,fake_output):
    real_loss = loss(tf.ones_like(real_output),real_output)
    fake_loss = loss(tf.zeros_like(fake_output),fake_output)
    total = fake_loss+real_loss
    return total

#Calculates the loss based on discriminator false positives
def gen_loss(fake_output):
    #the first input is the ground truth, which would be all ones
    #if the generator did well
    #the second input is the dicriminator's predictions
    return loss(tf.ones_like(fake_output),fake_output)
