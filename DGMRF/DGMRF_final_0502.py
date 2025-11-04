#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np


# In[2]:


def load_atlas_csv(file_path):
    """Load atlas image from CSV into TensorFlow tensor (1 x H x W x 1)"""
    image = np.loadtxt(file_path, delimiter=',').astype(np.float32)
    image_tensor = tf.convert_to_tensor(image[None, ..., None])
    return image_tensor


# In[3]:


# 2️⃣ DeepGMRF_QExtractor 클래스
class DeepGMRF_QExtractor:
    def __init__(self, img_shape, n_layers=1, neighbor_mask=None, diagonal_loading_eps=1e-3, scaling_factor=1.0):
        self.H, self.W = img_shape
        self.n_layers = n_layers
        self.diagonal_loading_eps = diagonal_loading_eps
        self.scaling_factor = scaling_factor
        
        if neighbor_mask is None:
            self.neighbor_mask = np.array([[0,1,0],
                                           [1,0,1],
                                           [0,1,0]])
        else:
            self.neighbor_mask = neighbor_mask
            
        # filter weights (trainable TensorFlow Variables)
        self.filter_weights = {}
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                if self.neighbor_mask[dx+1, dy+1]:
                    key = (dx, dy)
                    self.filter_weights[key] = tf.Variable(tf.random.normal([1], stddev=0.1), trainable=True)

    def cnn_convolve(self, x):
        """Apply manual convolution with learned filter weights"""
        output = tf.zeros_like(x)
        for (dx, dy), w_var in self.filter_weights.items():
            shifted = tf.roll(x, shift=[dx, dy], axis=[1,2])
            output += w_var * shifted
        return output

    def get_sparse_operator_tensorflow(self):
        """Create sparse operator as dense TensorFlow matrix (for gradient flow)"""
        row_indices = []
        col_indices = []
        values = []
        
        idx_map = lambda x, y: x * self.W + y
        
        for i in range(self.H):
            for j in range(self.W):
                center_idx = idx_map(i, j)
                diag_sum = tf.constant(0.0, dtype=tf.float32)
                
                for (dx, dy), w_var in self.filter_weights.items():
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < self.H and 0 <= nj < self.W:
                        neighbor_idx = idx_map(ni, nj)
                        row_indices.append(center_idx)
                        col_indices.append(neighbor_idx)
                        values.append(-w_var[0])
                        diag_sum += w_var[0]
                        
                row_indices.append(center_idx)
                col_indices.append(center_idx)
                values.append(diag_sum)
        
        indices = tf.stack([tf.convert_to_tensor(row_indices, dtype=tf.int64),
                            tf.convert_to_tensor(col_indices, dtype=tf.int64)], axis=1)
        values = tf.stack(values)
        shape = [self.H * self.W, self.H * self.W]
        
        L_dense = tf.scatter_nd(indices, values, shape)
        return L_dense

    def get_precision_matrix(self):
        """Compute Q = L Lᵀ + safeguard"""
        L = self.get_sparse_operator_tensorflow()
        L_stack = L
        for _ in range(self.n_layers - 1):
            L_stack = tf.matmul(L_stack, L)
        Q = tf.matmul(L_stack, tf.transpose(L_stack))
        
        # Ensure positive definiteness
        eigvals = tf.linalg.eigvalsh(Q)
        min_eig = tf.reduce_min(eigvals)
        
        tf.print("Minimum eigenvalue before correction:", min_eig)
        
        def apply_diag_loading(Q, delta):
            return Q + tf.eye(Q.shape[0]) * delta
        
        Q_corrected = tf.cond(min_eig <= 0,
                              lambda: apply_diag_loading(Q, tf.abs(min_eig) + self.diagonal_loading_eps),
                              lambda: Q)
        
        if self.scaling_factor != 1.0:
            Q_corrected = Q_corrected * self.scaling_factor
        
        eigvals_post = tf.linalg.eigvalsh(Q_corrected)
        tf.print("Eigenvalue range after correction:", tf.reduce_min(eigvals_post), tf.reduce_max(eigvals_post))
        
        return Q_corrected


# In[4]:


# 3️⃣ CNN filter 학습
def train_deepgmrf(atlas_list, extractor, num_epochs=100, learning_rate=0.01):
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            total_loss = 0.0
            for atlas in atlas_list:
                pred = extractor.cnn_convolve(atlas)
                reconstruction = tf.reduce_mean(tf.square(pred - atlas))
                latent_reg = tf.reduce_mean(tf.square(pred))
                loss = 0.5 * reconstruction + 0.5 * latent_reg
                total_loss += loss
            avg_loss = total_loss / len(atlas_list)
        
        grads = tape.gradient(avg_loss, extractor.filter_weights.values())
        optimizer.apply_gradients(zip(grads, extractor.filter_weights.values()))
        
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            tf.print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")


# In[162]:


# 1️⃣ atlas 이미지 불러오기
atlas1 = load_atlas_csv('/Users/boyoung/Desktop/Research_moment/Data_ADNI/9 parcels/atlas1_p9.csv')
atlas2 = load_atlas_csv('/Users/boyoung/Desktop/Research_moment/Data_ADNI/9 parcels/atlas2_p9.csv')
atlas3 = load_atlas_csv('/Users/boyoung/Desktop/Research_moment/Data_ADNI/9 parcels/atlas3_p9.csv')
atlas4 = load_atlas_csv('/Users/boyoung/Desktop/Research_moment/Data_ADNI/9 parcels/atlas4_p9.csv')
atlas5 = load_atlas_csv('/Users/boyoung/Desktop/Research_moment/Data_ADNI/9 parcels/atlas5_p9.csv')
atlas6 = load_atlas_csv('/Users/boyoung/Desktop/Research_moment/Data_ADNI/9 parcels/atlas6_p9.csv')
atlas_list = [atlas1, atlas2, atlas3, atlas4, atlas5, atlas6]


# In[6]:


# 1️⃣ atlas 이미지 불러오기 (Simulation data)
atlas1 = load_atlas_csv('/Users/boyoung/Desktop/Research_moment/bdatls1.csv')
atlas2 = load_atlas_csv('/Users/boyoung/Desktop/Research_moment/bdatls2.csv')
atlas3 = load_atlas_csv('/Users/boyoung/Desktop/Research_moment/bdatls3.csv')
atlas4 = load_atlas_csv('/Users/boyoung/Desktop/Research_moment/gdatls.csv')

atlas_list = [atlas1, atlas2, atlas3, atlas4]


# In[33]:


# 2️⃣ CNN filter 학습
H = 40
W = 40
n_layers = 6


# In[34]:


extractor = DeepGMRF_QExtractor(img_shape=(H,W), n_layers=n_layers)
train_deepgmrf(atlas_list, extractor, num_epochs=100, learning_rate=0.01)


# In[35]:


# 3️⃣ Q 추출
Q = extractor.get_precision_matrix()
print("Q matrix shape:", Q.shape)


# In[36]:


print(Q)


# In[37]:


np.savetxt("/Users/boyoung/Desktop/Research_moment/DGMRF_simulation/atlas_parcels_csv/Q_L6/Q_total.csv", Q , delimiter=",", fmt="%.6f")


# In[ ]:




