```python
import tensorflow as tf
```


```python
#creating nodes in computational graph
node1=tf.constant(3,dtype=tf.int32)
node2=tf.constant(5,dtype=tf.int32)
node3=tf.add(node1,node2)
```


```python
tf.compat.v1.disable_eager_execution()
#create a tf session obj
sess = tf.compat.v1.Session()

```


```python
print("sum of  node1 and node2 is: ",sess.run(node3))
```

    sum of  node1 and node2 is:  8
    


```python
sess.close()
```


```python
#example using variBLE
```


```python
import tensorflow as tf
```


```python
node=tf.Variable(tf.zeros([2,2]))
node
```




    <tf.Variable 'Variable:0' shape=(2, 2) dtype=float32>




```python
tf.compat.v1.disable_eager_execution()
tf.compat.v1.global_variables_initializer()

#create a tf session obj
#sess = tf.compat.v1.Session()

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    print("tensor val before addn:\n",sess.run(node))
    ewa=node.assign(node+tf.ones([2,2]))
    print("tensor val after addn:\n",sess.run(ewa))
```

    tensor val before addn:
     [[0. 0.]
     [0. 0.]]
    tensor val after addn:
     [[1. 1.]
     [1. 1.]]
    


```python
#placeholders
#A graph can be parametrized to accept external inputs known
#as placeholders
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
```


```python
a=tf.placeholder(tf.int32,shape=(3,1))
b=tf.placeholder(tf.int32,shape=(1,3))
c=tf.matmul(a,b)
```


```python
with tf.compat.v1.Session() as sess:
    print(sess.run(c,feed_dict={a:[[3],[2],[1]],b:[[1,2,3]]}))
```

    [[3 6 9]
     [2 4 6]
     [1 2 3]]
    


```python
tf.__version__

```




    '2.4.1'




```python

```
