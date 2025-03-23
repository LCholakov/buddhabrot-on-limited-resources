import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from scipy.ndimage import gaussian_filter


def coord_to_pixel(x, y, width, height, xmin, xmax, ymin, ymax):
    """Convert complex coordinates to pixel indices."""
    px = int((x - xmin) / (xmax - xmin) * width)
    py = int((y - ymin) / (ymax - ymin) * height)
    return px, py


def buddhabrot_worker(args):
    """
    Worker function to generate a portion of the Buddhabrot accumulation array.
    """
    samples, width, height, xmin, xmax, ymin, ymax, max_iter = args
    acc = np.zeros((height, width), dtype=np.float64)
    for _ in range(samples):
        c_real = np.random.uniform(xmin, xmax)
        c_imag = np.random.uniform(ymin, ymax)
        c = complex(c_real, c_imag)
        z = 0 + 0j
        orbit = []
        for i in range(max_iter):
            z = z * z + c
            orbit.append(z)
            # Check escape condition (|z| > 2)
            if (z.real * z.real + z.imag * z.imag) > 4.0:
                for point in orbit:
                    x_val, y_val = point.real, point.imag
                    if xmin <= x_val <= xmax and ymin <= y_val <= ymax:
                        px, py = coord_to_pixel(x_val, y_val, width, height, xmin, xmax, ymin, ymax)
                        if 0 <= px < width and 0 <= py < height:
                            acc[py, px] += 1
                break
    return acc


def main():
    # High-resolution image parameters (4K resolution example)
    width, height = 3840, 2160
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5
    max_iter = 5000

    # Increase sample count for smoother results
    # total_samples = 5000000
    total_samples = 3000000
    num_processes = cpu_count()
    samples_per_process = total_samples // num_processes

    args_list = [
        (samples_per_process, width, height, xmin, xmax, ymin, ymax, max_iter)
        for _ in range(num_processes)
    ]

    print(f"Using {num_processes} processes with {samples_per_process} samples each...")

    # Parallel computation of the accumulation array.
    with Pool(num_processes) as pool:
        results = pool.map(buddhabrot_worker, args_list)

    # Sum up the results from all processes.
    acc = np.sum(results, axis=0)

    # Apply logarithmic scaling for dynamic range compression.
    acc = np.log(acc + 1)

    # Apply a Gaussian filter to smooth the image.
    acc_smooth = gaussian_filter(acc, sigma=1)

    # Rotate the smoothed accumulation array 90° clockwise.
    # np.rot90 with k=-1 rotates the array 90° clockwise.
    acc_smooth_rot = np.rot90(acc_smooth)
    # Adjust the extent so that:
    # - The new horizontal axis represents the imaginary part (from ymax to ymin).
    # - The new vertical axis represents the real part (from xmin to xmax).
    new_extent = [ymax, ymin, xmin, xmax]

    # Plot the rotated image.
    plt.figure(figsize=(16, 9))
    plt.imshow(acc_smooth_rot, extent=new_extent, cmap='inferno',
               origin='lower', interpolation='bilinear')
    plt.title('High Resolution Buddhabrot Rotated 90° Clockwise')
    plt.xlabel('Imaginary Axis')
    plt.ylabel('Real Axis')
    plt.show()


if __name__ == '__main__':
    main()