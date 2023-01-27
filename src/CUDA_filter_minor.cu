#ifdef CUDA
/**
 * @file CUDA_filter_minor.cu
 * Routines for MINOR iltering of a color image
 * Multithreaded version using CUDA
 */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
#include "math_constants.h"
#include "device_functions.h"

#include <stdio.h>

#include "image.h"

 /**
  * @brief Implements the MINOR algorithm
  *
  * @param[in] in_img Image pointer { rgb }
  * @param[in] block_size Dimension of the Block { positive-odd }
  * @param[in] alpha Number of pixels taked into account { positive }
  * @param[in] beta Number of pixels taked into account { positive }
  * @param[in] sigma Sigma prameter { positive }
  *
  * @return Pointer to the filtered image or NULL
  *
  * @ref 1) TO be published 
  *
  * @author Kusnik Damian
  * @date 27.01.2023
  */
  
#define BLOCK_SIZE 4
#define ROUNDS 16

__global__
void normalizePixels(float* out_data, int* int_out_data, float* weight_data,int width, int data_len) {
	int ic = blockIdx.y * blockDim.y + threadIdx.y;
	int ir = blockIdx.x * blockDim.x + threadIdx.x;

	const unsigned int pos = ir * width + ic;
	if (ic >= width || pos >= data_len)
		return;
	int_out_data[pos] = ((int)(out_data[pos] / weight_data[pos]) << 16) |
		((int)(out_data[data_len + pos] / weight_data[pos]) << 8) |
		((int)(out_data[2 * data_len + pos] / weight_data[pos]));
	//pixel_data[2*data_len+pos] /= divider[pos]; //normalizeFactor[offset+x];
}

__device__
float get_alpha_nearest(int width, int height, int* in_data, int p, int qx, int qy, int f, int alpha)
{
	float sums[49];
	float sum = 0;
	float r, g, b;
	int k = 0;

	for (int i = -f; i <= f; i++) {
		if (qy + i < 0 || qy + i >= height) continue;
		for (int j = -f; j <= f; j++) {
			if (qx + j < 0 || qx + j >= width) continue;

			int q = (qy + i) * width + (qx + j);

			r = ((in_data[p] & 0xFF0000) >> 16) - ((in_data[q] & 0xFF0000) >> 16);
			g = ((in_data[p] & 0xFF00) >> 8) - ((in_data[q] & 0xFF00) >> 8);
			b = ((in_data[p] & 0xFF) - (in_data[q] & 0xFF));
			sums[k] = r * r + g * g + b * b;
			k++;
		}
	}
	int i_max = k;
	
	for (int i = 0; (i < alpha) && (i < i_max); i++)
	{
		float min = sums[0];
		int tmp = 0;
		for (int j = 1; j < i_max; j++)
		{
			if (sums[j] < min)
			{
				min = sums[j];
				tmp = j;
			}
		}
		sum += min;
		sums[tmp] = +INFINITY;
	}
	return sum;
}


__device__
void patch_weights(int width, int height, int* in_data, float* out_data, float* weights, int px, int py, int qx, int qy, int f, int alpha, int beta, float sigma)
{
	unsigned int data_len = height * width;
	float sum = 0.0;
	float sums[49];
	int p_sum[49];
	int q_sum[49];
	int p_min[49];
	int q_min[49];
	float w = 0.0;
	
	int k = 0;
	for (int i = -f; i <= f; i++) {
		if (py + i < 0 || py + i >= height) continue;
		if (qy + i < 0 || qy + i >= height) continue;
		for (int j = -f; j <= f; j++) {
			if (px + j < 0 || px + j >= width) continue;
			if (qx + j < 0 || qx + j >= width) continue;

			int p = (py + i) * width + (px + j);
			int q = (qy + i) * width + (qx + j);

			sums[k] = get_alpha_nearest(width, height, in_data, q, px, py, f, alpha);
			p_sum[k] = p;
			q_sum[k] = q;
			k++;
		}
	}
	int i_max = k;
	
	for (int i = 0; (i < beta) && (i<i_max); i++)
	{
		float min = sums[0];
		int tmp = 0;
		for (int j = 1; j < i_max; j++)
		{
			if (sums[j] < min)
			{
				min = sums[j];
				tmp = j;
			}
		}
		p_min[i] = p_sum[tmp];
		q_min[i] = q_sum[tmp];
		sum += min;
		sums[tmp] = INFINITY;
	}
	w = __expf(-(sum / (alpha * beta)/sigma));

	for (int i = 0; (i < beta )&& (i < i_max); i++)
	{
		atomicAdd(&out_data[p_min[i]], ((in_data[q_min[i]] & 0xFF0000) >> 16)* w);
		atomicAdd(&out_data[data_len + p_min[i]], ((in_data[q_min[i]] & 0xFF00) >> 8)*w);
		atomicAdd(&out_data[data_len + data_len + p_min[i]], (in_data[q_min[i]] & 0xFF)*w);
		atomicAdd(&weights[p_min[i]], w);
	}
	
	return;
}

__global__
void denoise_patch(int* in_data, float* out_data, float* weights, const int width, const int height, const int r, const int f, const int alpha, const int beta, const float sigma)
{
	int ic = blockIdx.y * blockDim.y + threadIdx.y;
	int ir = blockIdx.x * blockDim.x + threadIdx.x;
	if (ic >= width || ir >= height)
		return;

	int istart = max(ir - r, 0);
	int iend = min(ir + r, height - 1);
	int jstart = max(ic - r, 0);
	int jend = min(ic + r, width - 1);

	// go through all patches
	for (int i = istart; i <= iend; i++) { // i = y
		for (int j = jstart; j <= jend; j++) { // j = x
			patch_weights(width, height, in_data, out_data, weights, ic, ir, j, i, f, alpha, beta, sigma);
		}
	}
	return;
}
Image*
filter_minor(const Image* in_img, const int r, const int f, const int alpha, const int beta, const float sigma)
{
	float elapsed_time;
	clock_t start_time;
	start_time = start_timer();
	
	SET_FUNC_NAME("filter_minor");
	byte*** in_data;
	byte*** out_data;
	int num_rows, num_cols;
	int half_win, half_block;
	Image* out_img;

	if (!is_rgb_img(in_img))
	{
		ERROR_RET("Not a color image !", NULL);
	}

	if (!IS_POS(r))
	{
		ERROR("Window size ( %d ) must be positive!", r);
		return NULL;
	}

	if (!IS_POS(f))
	{
		ERROR("Alpha value ( %d ) must be positive !", f);
		return NULL;
	}

	if (!IS_POS(sigma))
	{
		ERROR("Sigma value ( %d ) must be positive !", sigma);
		return NULL;
	}

	num_rows = get_num_rows(in_img);
	num_cols = get_num_cols(in_img);
	int data_len = num_rows * num_cols;

	in_data = (byte***)get_img_data_nd(in_img);
	out_img = alloc_img(PIX_RGB, num_rows, num_cols);
	out_data = (byte***)get_img_data_nd(out_img);

//	cudaProfilerStart();


	size_t size_b = size_t(num_rows * num_cols) * sizeof(byte);
	size_t size_i = size_t(num_rows * num_cols) * sizeof(int);
	size_t size_f = size_t(num_rows * num_cols) * sizeof(float);

	int* int_in_data = (int*)malloc(size_i);
	for (int i = 0; i < num_rows; i++) {
		for (int j = 0; j < num_cols; j++)
		{
			int_in_data[i * num_cols + j] = (((int)in_data[i][j][0]) << 16) | ((int)in_data[i][j][1] << 8) | ((int)in_data[i][j][2]);
		}
	}
	
	int* d_int_out_data;
	cudaMalloc((void**)&d_int_out_data, size_i);

	float* d_out_data;
	cudaMalloc((void**)&d_out_data, size_f * 3);
	cudaMemset(d_out_data, 0, size_f*3);

	float* d_weight_data;
	cudaMalloc((void**)&d_weight_data, size_f);
	cudaMemset(d_weight_data, 0, size_f);

	int* d_in_data;
	cudaMalloc((void**)&d_in_data, size_i);
	cudaMemcpy(d_in_data, int_in_data, size_i, cudaMemcpyHostToDevice);

	dim3 blockDim(1, 64, 1);
	dim3 gridDim((unsigned int)ceil((float)num_rows / (float)blockDim.x),
		(unsigned int)ceil((float)num_cols / (float)blockDim.y),
		1);

        denoise_patch << < gridDim, blockDim >> > (d_in_data, d_out_data, d_weight_data, num_cols, num_rows, r, f, alpha, beta, sigma * sigma);
        cudaDeviceSynchronize();
        normalizePixels << < gridDim, blockDim >> > (d_out_data, d_int_out_data, d_weight_data, num_cols, num_cols * num_rows);
        cudaDeviceSynchronize();

	int* int_out_data = (int*)malloc(size_i);
	cudaMemcpy(int_out_data, d_int_out_data, size_i, cudaMemcpyDeviceToHost);


	for (int i = 0; i < num_rows; i++)
		for (int j = 0; j < num_cols; j++)
		{
			out_data[i][j][0] = (int_out_data[i * num_cols + j] >> 16) & 0xFF;
			out_data[i][j][1] = (int_out_data[i * num_cols + j] >> 8) & 0xFF;
			out_data[i][j][2] = (int_out_data[i * num_cols + j]) & 0xFF;

		}

	// Free device memory
	cudaFree(d_in_data);
	cudaFree(d_out_data);
	cudaFree(d_weight_data);
	cudaFree(d_int_out_data);
	cudaDeviceSynchronize();

	free(int_in_data);
	free(int_out_data);

	return out_img;
}
#endif
