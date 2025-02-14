from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

def load_image(path):
    image = np.array(Image.open(path))
    new_image = np.zeros((image.shape[0], image.shape[1]))

    print(f"Širina slike je > {image.shape[1]}px, a visina slike je > {image.shape[0]}px")

    if len(image.shape) > 2:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                # uprosjecivanje 
                # new_img[i][j] = int((image[i][j][0] + image[i][j][1] + image[i][j][2])/3)
                new_image[i][j] = int((0.299 * image[i][j][0] + 0.587 * image[i][j][1] + 0.114 * image[i][j][2]))
        gray_image_arr = new_image.astype(int)
        gray_image = (Image.fromarray(gray_image_arr)).convert("RGB").save("gray_image.jpg")
        image = gray_image_arr

    max_v = 0
    min_v = 255
    for row in image:
        if max(row) > max_v:
            max_v = max(row)
        if min(row) < min_v:
            min_v = min(row)

    print(f"Maksimalna vrijednost inteziteta je > {max_v}, a minimalna je > {min_v}")

    sliced = image[:10, :10]
    print(f"Inteziteti lijevog odsjecka 10x10 > ")
    print(sliced)
    print(f"Original > {image.dtype}")
    image = image.astype(float)
    print(f"After > {image.dtype}")
    return image

def gaussian(image, sigma):
    sigma_arr = [1,2,3,4,5]

    for s in sigma_arr:
        gauss_image = gaussian_filter(image,sigma=s)
        name = "gaussian" + str(s) + ".jpg"
        gauss_image = (Image.fromarray(gauss_image)).convert("RGB").save(name)
    
    gaussian_image_arr = gaussian_filter(image,sigma)

    return gaussian_image_arr

def normalize_image(img):
    min_val = img.min()
    max_val = img.max()

    if max_val - min_val == 0:
        return np.zeros_like(img)

    normalized = 255 * (img - min_val) / (max_val - min_val)
    return normalized.astype(np.uint8)

def calculate_gradient(image, Ix, Iy):

    Ix = ndimage.convolve(image, Ix)
    Iy = ndimage.convolve(image, Iy)

    image_Ix_normalized = normalize_image(Ix)
    image_Iy_normalized = normalize_image(Iy)

    name = "gradijent_Ix.jpg"
    gradient_image = (Image.fromarray(image_Ix_normalized)).convert("RGB").save(name)
    name = "gradijent_Iy.jpg"
    gradient_image = (Image.fromarray(image_Iy_normalized)).convert("RGB").save(name)

    Ix2 = Ix ** 2
    image_Ix2_normalized = normalize_image(Ix2)
    name = "gradijent_Ix2.jpg"
    gradient_image = (Image.fromarray(image_Ix2_normalized)).convert("RGB").save(name)

    IxIy = Ix * Iy
    image_IxIy_normalized = normalize_image(IxIy)
    name = "gradijent_IxIy.jpg"
    gradient_image = (Image.fromarray(image_IxIy_normalized)).convert("RGB").save(name)

    Iy2 = Iy ** 2
    image_Iy2_normalized = normalize_image(Iy2)
    name = "gradijent_Iy2.jpg"
    gradient_image = (Image.fromarray(image_Iy2_normalized)).convert("RGB").save(name)

    return Ix, Iy, Ix2, Iy2, IxIy

def gradient_sum(kernel_size, image_Ix2, image_Iy2, image_IxIy):
    kernel = np.ones((kernel_size, kernel_size), dtype=float) / (kernel_size ** 2)
    Sx2 = ndimage.convolve(image_Ix2, kernel).astype(float)
    Sy2 = ndimage.convolve(image_Iy2, kernel).astype(float)
    SxSy = ndimage.convolve(image_IxIy, kernel).astype(float)

    M = [[Sx2,SxSy],[Sy2,SxSy]]

    return M, Sx2, Sy2, SxSy

def harris_response(k, Sx2, Sy2, SxSy):

    det_M = (Sx2 * Sy2) - (SxSy ** 2)
    trace_M = Sx2 + Sy2
    R = det_M - k * (trace_M ** 2)
    return R

def harris_suppresion(threshold, harris_response, neighborhood_size):

    harris_response[harris_response < threshold] = 0

    nms_response = np.zeros_like(harris_response)
    neighborhood_size = 32

    half_size = neighborhood_size // 2
    for i in range(half_size, harris_response.shape[0] - half_size):
        for j in range(half_size, harris_response.shape[1] - half_size):
            neigh = harris_response[i - half_size:i + half_size + 1, j - half_size:j + half_size + 1]
            if harris_response[i, j] == neigh.max():
                nms_response[i, j] = harris_response[i, j]

    new_image = (Image.fromarray(nms_response)).convert("RGB").save("zadatak6.jpg")

    return nms_response

def k_nn(k, nms_response):
    non_zero_coords = np.nonzero(nms_response)
    non_zero_responses = nms_response[non_zero_coords]

    if k > len(non_zero_responses):
        print(f"Warning: Requested top {k} responses, but only {len(non_zero_responses)} non-zero responses found.")
        k = len(non_zero_responses)
    
    top_k_indices = np.argpartition(non_zero_responses, -k)[-k:]
    top_k_indices = top_k_indices[np.argsort(non_zero_responses[top_k_indices])[::-1]]
    top_k_coords = [(non_zero_coords[0][i], non_zero_coords[1][i], non_zero_responses[i]) for i in top_k_indices]
    
    for coord in top_k_coords:
        x, y, intensity = coord

    image = Image.open(path)
    plt.imshow(image,cmap="gray")

    for (y, x, value) in top_k_coords:
        circle = Circle((x, y), radius=3, color='red', fill=False)
        plt.gca().add_patch(circle)

    plt.title("Top K Harris Corners")
    plt.show()

def calculate_gradient_direction(Gx, Gy):

    magnitude = np.sqrt(Gx**2 + Gy**2)
    magnitude = (magnitude / magnitude.max()) * 255

    direction = np.arctan2(Gy,Gx)

    return magnitude, direction

def non_maximum_suppression(magnitude, direction):

    nms_output = np.zeros_like(magnitude)

    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):

            theta_angel = direction[i, j]

            if theta_angel < 22.5 and theta_angel > -22.5:
                if magnitude[i, j] >= magnitude[i, j - 1] and magnitude[i, j] >= magnitude[i, j + 1]:
                    nms_output[i, j] = magnitude[i, j]

            elif theta_angel > 22.5 and theta_angel > 67.5:
                if magnitude[i, j] >= magnitude[i - 1, j + 1] and magnitude[i, j] >= magnitude[i + 1, j - 1]:
                    nms_output[i, j] = magnitude[i, j]

            elif theta_angel > 67.5 and theta_angel < 112.5:
                if magnitude[i, j] >= magnitude[i - 1, j] and magnitude[i, j] >= magnitude[i + 1, j]:
                    nms_output[i, j] = magnitude[i, j]

            elif theta_angel > 112.5 and theta_angel < 157.5:
                if magnitude[i, j] >= magnitude[i - 1, j - 1] and magnitude[i, j] >= magnitude[i + 1, j + 1]:
                    nms_output[i, j] = magnitude[i, j]

    return nms_output

def hysteresis(nms_output, low_thresh, high_thresh, include_weak):
    canny_edges = np.zeros_like(nms_output)

    strong_edges = np.greater_equal(nms_output, high_thresh)
    weak_edges = np.greater_equal(nms_output, low_thresh) & np.less(nms_output, high_thresh)

    strong, weak = 255, 100
    canny_edges[strong_edges] = strong
    canny_edges[weak_edges] = weak

    for i in range(1, nms_output.shape[0] - 1):
        for j in range(1, nms_output.shape[1] - 1):
            neighbor_state = False
            koordinate = [-1, 1 ,2]
            for k in koordinate:
                for l in koordinate:
                    new_i = i + k
                    new_j = j + l
                    if 0 <= new_i < canny_edges.shape[0] and 0 <= new_j < canny_edges.shape[1]:
                        if canny_edges[new_i, new_j] == strong:
                            neighbor_state = True
            if include_weak:
                if canny_edges[i, j] == weak and neighbor_state:
                    canny_edges[i, j] = strong
                elif canny_edges[i, j] == weak:
                    canny_edges[i, j] = weak
                neighbor_state = False
            else:
                if canny_edges[i, j] == weak and neighbor_state:
                    canny_edges[i, j] = strong
                elif canny_edges[i, j] == weak:
                    canny_edges[i, j] = 0
                neighbor_state = False
    return canny_edges

zadatak = 2

if zadatak == 1:
    # zadatak 1.1
    path = "fer.jpg"
    image_array = load_image(path=path)

    #zadatak 1.2
    sigma = 1
    image_array = gaussian(image_array, sigma)

    #zadatak 1.3
    Ix = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    Iy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

    Ix, Iy, Ix2, Iy2, IxIy = calculate_gradient(image_array, Ix,Iy)

    #zadatak 1.4
    kernel_size = 5
    M, Sx2, Sy2, SxSy = gradient_sum(kernel_size, Ix2, Iy2, IxIy)

    #zadatak 1.5
    k = 0.04
    harris_response_array = harris_response(k, Sx2, Sy2, SxSy)

    print(harris_response_array.max())
    print(harris_response_array.min())

    name = "harris_response_image.jpg"
    harris_response_image = (Image.fromarray(harris_response_array)).convert("RGB").save(name)
    #zadatak 1.6

    threshold = 1e9
    neighborhood_size = 14
    nms_response = harris_suppresion(threshold, harris_response_array, neighborhood_size)

    #zadatak 1.7

    k = 100
    k_nn(k, nms_response)

else:

    #zadatak 2.1
    path = "house.jpg"
    image_array = load_image(path=path)

    #zadatak 2.2
    sigma = 1.5
    image_array = gaussian(image_array, sigma)

    #zadatak 2.3
    Ix = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    Iy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

    image_Ix, image_Iy, image_array_Ix2, image_array_Iy2, image_array_IxIy = calculate_gradient(image_array, Ix,Iy)

    #zadatak 2.4
    magnitude, direction = calculate_gradient_direction(image_Ix, image_Iy)
    name = "magnitude_image.jpg"
    magnitude_image = (Image.fromarray(magnitude)).convert("RGB").save(name)

    #zadatak 2.5
    nms_output = non_maximum_suppression(magnitude, direction)
    name = "nms.jpg"
    nms_image = (Image.fromarray(nms_output)).convert("RGB").save(name)

    # zadatak 2.6
    low, high = 10, 90
    include_weak = False
    canny_edge = hysteresis(nms_output, low, high,include_weak)
    name = "final_canny.jpg"
    nms_image = (Image.fromarray(canny_edge)).convert("RGB").save(name)
