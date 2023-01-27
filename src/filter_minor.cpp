#ifndef CUDA
#include <stdio.h>
#include "image.h"
 /**
  * @brief Implements the MINOR algorithm
  *
  * @param[in] in_img Image pointer { rgb }
  * @param[in] block_size Dimension of the Block { positive-odd }
  * @param[in] alpha Number of pixels taked into account { positive }
  * @param[in] beta Number of pixels taked from alpha averaged { positive }
  * @param[in] sigma Sigma prameter { positive }
  * @param[in] f patch size { positive }
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

void normalizePixels(float* out_data, int* int_out_data, float* weight_data,int width, int data_len, int ic, int ir)
{
	const unsigned int pos = ir * width + ic;
	if (ic >= width || pos >= data_len)
		return;
	int_out_data[pos] = ((int)(out_data[pos] / weight_data[pos]) << 16) |
		((int)(out_data[data_len + pos] / weight_data[pos]) << 8) |
		((int)(out_data[2 * data_len + pos] / weight_data[pos]));
}

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

void patch_weights(int width, int height, int* in_data, float* out_data, float* weights, int px, int py, int qx, int qy, int f, int alpha, int beta, float sigma)
{
	unsigned int data_len = height * width;
	float sum = 0.0;
	float sums[49]; // weight,x,y
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
	w = expf(-(sum / (alpha * beta)/sigma));
	
	for (int i = 0; (i < beta )&& (i < i_max); i++)
	{
        out_data[p_min[i]] += ((in_data[q_min[i]] & 0xFF0000) >> 16) * w;
		out_data[data_len + p_min[i]] += ((in_data[q_min[i]] & 0xFF00) >> 8) * w;
		out_data[data_len + data_len + p_min[i]] += (float) (in_data[q_min[i]] & 0xFF) * w;

		weights[p_min[i]] += w;
	}
	
	return;
}

void denoise_patch(int* in_data, float* out_data, float* weights, const int width, const int height, const int r, const int f, const int alpha, const int beta, const float sigma,
        int ic, int ir)
{
	if (ic >= width || ir >= height)
		return;

	int istart = MAX(ir - r, 0);
	int iend = MIN(ir + r, height - 1);
	int jstart = MAX(ic - r, 0);
	int jend = MIN(ic + r, width - 1);

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

	size_t size_b = size_t(num_rows * num_cols) * sizeof(byte);
	size_t size_i = size_t(num_rows * num_cols) * sizeof(int);
	size_t size_f = size_t(num_rows * num_cols) * sizeof(float);

	int* int_in_data = (int*)malloc(size_i);
    float* float_out_data = (float*)malloc(3 * size_f);	
    float* weight_data = (float*)malloc(size_f);	
	int* int_out_data = (int*)malloc(size_i);

	for (int i = 0; i < num_rows; i++) {
		for (int j = 0; j < num_cols; j++)
		{
			int_in_data[i * num_cols + j] = (((int)in_data[i][j][0]) << 16) | ((int)in_data[i][j][1] << 8) | ((int)in_data[i][j][2]);
			weight_data[i * num_cols + j] = 0;
			float_out_data[i * num_cols + j] = 0;
			float_out_data[(i * num_cols + j) * 2] = 0;
			float_out_data[(i * num_cols + j) * 3] = 0;
		}
	}

    start_time = start_timer();
#pragma omp parallel \
     shared(int_in_data, float_out_data, weight_data)
{
#pragma omp for schedule(dynamic) nowait
for (int ir = 0; ir < num_rows; ir++)
    for (int ic = 0; ic < num_cols; ic++)
        denoise_patch(int_in_data, float_out_data, weight_data, num_cols, num_rows, r, f, alpha, beta, sigma * sigma, ic, ir);
}
#pragma omp parallel \
     shared(float_out_data, int_out_data, weight_data)
{
#pragma omp for schedule(dynamic) nowait
for (int ir = 0; ir < num_rows; ir++)
    for (int ic = 0; ic < num_cols; ic++)
        normalizePixels(float_out_data, int_out_data, weight_data, num_cols, num_cols * num_rows, ic, ir);
}

	for (int i = 0; i < num_rows; i++)
		for (int j = 0; j < num_cols; j++)
		{
			out_data[i][j][0] = (int_out_data[i * num_cols + j] >> 16) & 0xFF;
			out_data[i][j][1] = (int_out_data[i * num_cols + j] >> 8) & 0xFF;
			out_data[i][j][2] = (int_out_data[i * num_cols + j]) & 0xFF;

		}

	// Free device memory
	free(float_out_data);
	free(weight_data);
	free(int_in_data);
	free(int_out_data);

	return out_img;
}
#endif
