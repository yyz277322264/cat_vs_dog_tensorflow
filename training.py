import os
import numpy as np
import tensorflow as tf
import input_data
import model

N_CLASSES=30
IMG_H=256
IMG_W=256
BATCH_SIZE=16
CAPACITY=2000
MAX_STEP=15000
learning_rate=0.0001

def run_training():
    train_dir = 'E:\yyz\python\Pig_rec/train/'
    logs_train_dir='E:\yyz\python\Pig_rec/'
    train,train_label=input_data.get_files(train_dir)
    # print(len(train))
    # print(len(train_label))
    train_batch,train_label_batch=input_data.get_batch(train,
                                                       train_label,
                                                       IMG_W,
                                                       IMG_H,
                                                       BATCH_SIZE,
                                                       CAPACITY)
    # print(train_batch.shape)
    # (16, 208, 208, 3) (batch_size,img_w,img_h,nchannels)
    train_logits=model.inference(train_batch,BATCH_SIZE,N_CLASSES)
    # print(train_logits.shape)
    # (16, 2)
    train_loss=model.losses(train_logits,train_label_batch)
    train_op=model.training(train_loss,learning_rate)
    train_acc=model.evaluation(train_logits,train_label_batch)
    summary_op=tf.summary.merge_all()
    sess=tf.Session()
    train_writer=tf.summary.FileWriter(logs_train_dir,sess.graph)
    saver=tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _,tra_loss,tra_acc=sess.run([train_op,train_loss,train_acc])
            if step%50==0:
                print('Step %d,train loss= %.2f, train_acc=%.2f%%'%(step,tra_loss,tra_acc))
                summary_str=sess.run(summary_op)
                train_writer.add_summary(summary_str,step)
            if step%2000==0 or (step+1)==MAX_STEP:
                checkpoint_path=os.path.join(logs_train_dir,'model.ckpt')
                saver.save(sess,checkpoint_path,global_step=step)
    except tf.errors.OutOfRangeError:
        print('Done training!')
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()
if __name__ == '__main__':
    run_training()
