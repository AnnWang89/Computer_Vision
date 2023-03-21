import cv2
import numpy as np
import scipy
import open3d as o3d
import matplotlib.pyplot as plt

image_row = 120
image_col = 120


# visualizing the mask (size : "image width" * "image height")
def mask_visualization(M):
    mask = np.copy(np.reshape(M, (image_row, image_col)))
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')

# visualizing the unit normal vector in RGB color space
# N is the normal map which contains the "unit normal vector" of all pixels (size : "image width" * "image height" * 3)


def normal_visualization(N):
    # converting the array shape to (w*h) * 3 , every row is a normal vetor of one pixel
    N_map = np.copy(np.reshape(N, (image_row, image_col, 3)))
    # Rescale to [0,1] float number
    N_map = (N_map + 1.0) / 2.0
    plt.figure()
    print(N_map.shape)
    plt.imshow(N_map)
    plt.title('Normal map')

# visualizing the depth on 2D image
# D is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")


def depth_visualization(D):
    D_map = np.copy(np.reshape(D, (image_row, image_col)))
    # D = np.uint8(D)

    plt.figure()
    plt.imshow(D_map)
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth map')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')

# convert depth map to point cloud and save it to ply file
# Z is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")


def save_ply(Z, filepath):
    Z_map = np.reshape(Z, (image_row, image_col)).copy()
    data = np.zeros((image_row*image_col, 3), dtype=np.float32)
    # let all point float on a base plane
    baseline_val = np.min(Z_map)
    Z_map[np.where(Z_map == 0)] = baseline_val
    for i in range(image_row):
        for j in range(image_col):
            idx = i * image_col + j
            data[idx][0] = j
            data[idx][1] = i
            data[idx][2] = Z_map[image_row - 1 - i][j]
    # output to ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.io.write_point_cloud(filepath, pcd, write_ascii=True)

# show the result of saved ply file


def show_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])

# read the .bmp file


def read_bmp(filepath):
    global image_row
    global image_col
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    image_row, image_col = image.shape
    return image


def Light_Calculation(path):
    # light source
    light_lst = []
    with open(path + '/LightSource.txt', 'r', encoding='utf8') as shadow_file:
        for x in shadow_file.readlines():
            temp = x[7:-2].split(',')
            temp = [float(i) for i in temp]
            light_lst.append(temp)
    light_lst = np.array(light_lst)
    # print("光源向量:\n",S_lst)
    # print(S_lst.shape)

    # light normal vector
    unitvector_light = []
    for i in light_lst:
        unitvector_light.append(i/np.linalg.norm(i))
    light_lst = np.array(unitvector_light)
    # print("?_i 光源單位向量:\n", light_lst)
    # print(light_lst.shape)
    return light_lst


def Normal_Calculation(path, light_lst):
    # Read Image
    IMG1 = read_bmp(path + '/pic1.bmp')
    IMG2 = read_bmp(path + '/pic2.bmp')
    IMG3 = read_bmp(path + '/pic3.bmp')
    IMG4 = read_bmp(path + '/pic4.bmp')
    IMG5 = read_bmp(path + '/pic5.bmp')
    IMG6 = read_bmp(path + '/pic6.bmp')

    N_lst = np.zeros((image_row, image_col, 3), dtype=np.float32)
    # print(N_lst.shape)
    for x in range(IMG1.shape[0]):
        for y in range(IMG1.shape[1]):
            I = np.array([
                IMG1[x][y],
                IMG2[x][y],
                IMG3[x][y],
                IMG4[x][y],
                IMG5[x][y],
                IMG6[x][y]
            ])

            # 計算KN
            KdN = np.dot(np.dot(np.linalg.inv(
                np.dot(light_lst.T, light_lst)), light_lst.T), I)
            KdN = KdN.T

            # 計算N = KdN/|KdN|
            KdN_gray = KdN[0]*0.0722+KdN[1]*0.7152+KdN[2]*0.2126  # 相對光亮度
            KdN_norm = np.linalg.norm(KdN)
            if KdN_norm == 0:
                continue
            N_lst[x][y] = KdN/KdN_norm

            # for k in range(3):
            #    if KdN_norm == 0:
            #        continue
            #    N_lst[x][y][k] = KdN[k]/KdN_norm

    # 控制在0到255間
    # N_lst = ((N_lst*0.5 + 0.5)*255).astype(np.uint8)
    normal_visualization(N_lst)
    # print("N_lst:\n", N_lst)

    return N_lst


def Create_Mask(N_lst):
    Mask = np.zeros((image_row, image_col), dtype=np.float32)
    # N_lst = KdN[0]*0.0722+KdN[1]*0.7152+KdN[2]*0.2126
    for x in range(Mask.shape[0]):
        for y in range(Mask.shape[1]):
            if np.linalg.norm(N_lst[x][y]) == 0:
                Mask[x][y] = 0
            else:
                Mask[x][y] = 1
    mask_visualization(Mask)
    return Mask


def Depth_Calculation(N_lst):
    # Create Mask
    Mask = Create_Mask(N_lst)

    # Sparse Matrix Index
    obj_index_row, obj_index_col = np.where(Mask != 0)
    num_pixel = np.size(obj_index_row)
    obj_index = np.zeros((image_row, image_col), dtype=np.float32)
    for idx in range(np.size(obj_index_row)):
        obj_index[obj_index_row[idx], obj_index_col[idx]] = idx

    M = scipy.sparse.lil_matrix((2*num_pixel, num_pixel))
    V = np.zeros((2*num_pixel, 1))

    for idx in range(num_pixel):
        row = obj_index_row[idx]
        col = obj_index_col[idx]

        # normal vector
        N = np.zeros((3), dtype=np.float32)
        N[0] = N_lst[row][col][0]
        N[1] = N_lst[row][col][1]
        N[2] = N_lst[row][col][2]

        # horizontal
        row_idx = idx*2
        if Mask[row, col+1]:
            idx_hor = int(obj_index[row, col+1])
            M[row_idx, idx] = -1
            M[row_idx, idx_hor] = 1
            if N[2] == 0:
                V[row_idx] = 0
            else:
                V[row_idx] = -N[0]/N[2]

        elif Mask[row, col-1]:
            idx_hor = int(obj_index[row, col-1])
            M[row_idx, idx_hor] = -1
            M[row_idx, idx] = 1
            if N[2] == 0:
                V[row_idx] = 0
            else:
                V[row_idx] = -N[0]/N[2]

        # vertical
        row_idx = idx*2+1
        if Mask[row+1, col]:
            idx_ver = int(obj_index[row+1, col])
            M[row_idx, idx] = 1
            M[row_idx, idx_ver] = -1
            if N[2] == 0:
                V[row_idx] = 0
            else:
                V[row_idx] = -N[1]/N[2]

        elif Mask[row-1, col]:
            idx_ver = int(obj_index[row-1, col])
            M[row_idx, idx_ver] = 1
            M[row_idx, idx] = -1
            if N[2] == 0:
                V[row_idx] = 0
            else:
                V[row_idx] = -N[1]/N[2]

    # solve equation
    MTM = M.T @ M
    MTV = M.T @ V
    z = scipy.sparse.linalg.spsolve(MTM, MTV)

    # mean and std
    std_z = np.std(z, ddof=1)
    mean_z = np.mean(z)
    # z_zscore = (z - mean_z) / std_z
    # outlier_ind = np.abs(z_zscore) > 10
    # z_min = np.min(z[~outlier_ind])
    # z_max = np.max(z[~outlier_ind])

    # create a same shape of mask
    Z = Mask.astype('float')
    for idx in range(num_pixel):
        row = obj_index_row[idx]
        col = obj_index_col[idx]
        # Z[row,col] = (z[idx] - z_min) / (z_max - z_min) * 255
        Z[row, col] = z[idx]

    depth_visualization(Z)
    return Z


def photometric_stereo(path):
    filepath = "./test"+path
    # Light Calculation
    light_lst = Light_Calculation(filepath)

    # Normal Calculation
    N_lst = Normal_Calculation(filepath, light_lst)

    # Depth Calculation
    Z = Depth_Calculation(N_lst)
    save_ply(Z, filepath + path+".ply")
    show_ply(filepath + path+".ply")


if __name__ == '__main__':
    path1 = "/bunny"
    path2 = "/star"
    path3 = "/venus"
    photometric_stereo(path1)
    photometric_stereo(path2)
    photometric_stereo(path3)
    # depth_visualization(Z)
    # save_ply(Z,filepath)
    # show_ply(filepath)

    # showing the windows of all visualization function
    plt.show()
