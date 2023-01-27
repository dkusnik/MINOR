#include "image.h"
int main(int argc, char** argv)
{
	float elapsed_time;
	clock_t start_time;
	Image* in_img;
	Image* noisy_img;
	Image* out_img;
	int f;
	int alpha;
	int beta;
	int r;
	float sigma;

	printf("argc: %d\n", argc);


	if (argc < 3)
	{
		printf("argc: %d\n", argc);
		fprintf(stderr, "Usage: %s <reference image { rgb }> <noisy image {rgb}> <block_radius> <alpha> <beta> <sigma> <f>\n", argv[0]);
		exit(EXIT_FAILURE);
	}
    if (argc == 8)
	{
		r = atoi(argv[3]);
		alpha = atoi(argv[4]);
		beta = atoi(argv[5]);
		sigma = atof(argv[6]);
		f = atoi(argv[7]);
	}
	else {
		r = 3;
		f = 1;
		alpha = 4;
		beta = 4;
		sigma = 40;
	}

	printf("Testing MINOR Filter...\n");
	/* Read the input image */
	in_img = read_img(argv[1]);
	noisy_img = read_img(argv[2]);


	/* Make sure it's an rgb image */
	if (is_gray_img(in_img))
	{
		fprintf(stderr, "Input image ( %s ) must not be grayscale !", argv[1]);
		exit(EXIT_FAILURE);
	}

	/* Start the timer */
    start_time = start_timer();
    out_img = filter_minor(noisy_img, r, f, alpha, beta, sigma);
    elapsed_time = stop_timer(start_time);

    write_img(out_img, "out.png", FMT_PNG);
    printf("r, alpha, beta, sigma, f, time: %d, %d, %d, %f, %d, %f -->\n", r, alpha, beta, sigma, f, elapsed_time);

    calculate_snr(in_img, out_img, NULL);
    calculate_ssim(in_img, out_img, NULL);
    printf("MINOR time = %f ms\n", elapsed_time);
    free_img(out_img);

	free_img(in_img);
	free_img(noisy_img);
	return EXIT_SUCCESS;
}
