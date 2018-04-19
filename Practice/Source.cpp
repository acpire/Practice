#include <stdio.h>
#include <vector>
#include <string>
#include <Windows.h>
#include <amp.h>
#include <stack>
#include <string>
#include <amp_math.h>
#include <amp_short_vectors.h>
#include <amp_graphics.h>
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\core\core.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\opencv.hpp"
#ifndef _DEBUG
#pragma comment(lib,"opencv_world331.lib")
#else
#pragma comment(lib,"opencv_world331d.lib")
#endif
using namespace cv;
HWND hMainWnd;
struct Image
{
	HBITMAP hBitmap;
	LONG width, height;
};
struct RGB {
	uchar r, g, b;
	bool operator ==(RGB s) {
		if (s.r == r && s.g == g && s.b == b)
			return true;
		else
			return false;
	}
	bool operator !=(RGB s) {
		if (s.r != r && s.g != g && s.b != b)
			return true;
		else
			return false;
	}
	bool operator <(RGB s) {
		if (s.r > r && s.g > g && s.b > b)
			return true;
		else
			return false;
	}
	void operator +=(RGB s) {
		r += s.r;
		g += s.g;
		b += s.b;
	}
};
struct allPositions {
	Point hh;
	Point hl;
	uint32_t median;
};
#define size_filter_main  7
#define size_filter_last  5
std::vector<Mat> allimage;
std::vector<std::vector<allPositions>> dataLinesOnAllImages;
std::vector<std::string> namesReads;
HFONT hfontMSSS;
HWND hButton[16];
HDC imageDC;
size_t images_length;
Image* images;
HBITMAP ConvertCVMatToBMP(cv::Mat& frame)
{
	auto convertOpenCVBitDepthToBits = [](const INT32 value) {
		auto regular = 0u;

		switch (value) {
		case CV_8U:
		case CV_8S:
			regular = 8u;
			break;

		case CV_16U:
		case CV_16S:
			regular = 16u;
			break;

		case CV_32S:
		case CV_32F:
			regular = 32u;
			break;

		case CV_64F:
			regular = 64u;
			break;

		default:
			regular = 0u;
			break;
		}

		return regular;
	};
	Mat output;
	resize(frame, output, Size(), 0.25, 0.25, CV_INTER_AREA);
	auto imageSize = output.size();

	if (imageSize.width && imageSize.height) {
		auto headerInfo = BITMAPINFOHEADER{};
		ZeroMemory(&headerInfo, sizeof(headerInfo));

		headerInfo.biSize = sizeof(headerInfo);
		headerInfo.biWidth = imageSize.width;
		headerInfo.biHeight = -(imageSize.height);
		headerInfo.biPlanes = 1;

		const auto bits = convertOpenCVBitDepthToBits(output.depth());
		headerInfo.biBitCount = output.channels() * bits;

		auto bitmapInfo = BITMAPINFO{};
		ZeroMemory(&bitmapInfo, sizeof(bitmapInfo));

		bitmapInfo.bmiHeader = headerInfo;
		bitmapInfo.bmiColors->rgbBlue = 0;
		bitmapInfo.bmiColors->rgbGreen = 0;
		bitmapInfo.bmiColors->rgbRed = 0;
		bitmapInfo.bmiColors->rgbReserved = 0;

		auto dc = GetDC(nullptr);
		assert(dc != nullptr && "Failure to get DC");

		auto bmp = CreateDIBitmap(dc,
			&headerInfo,
			CBM_INIT,
			output.data,
			&bitmapInfo,
			DIB_RGB_COLORS);
		assert(bmp != nullptr && "Failure creating bitmap from captured frame");

		DeleteDC(dc);
		return bmp;
	}

	return nullptr;
}
DWORD WINAPI CalculateCompare(LPVOID lpParam) {
#pragma omp parallel for
	for (int32_t i = 0; i < allimage.size(); i++) {
		allimage[i] = imread(namesReads[i]);
		Mat image(allimage[i].size(), allimage[i].type());
		RGB* ptr = (RGB*)image.data;
		RGB* ptr_im = (RGB*)allimage[i].data;
		InvalidateRect(hMainWnd, NULL, FALSE);
		size_t numberImages = 0;

		for (size_t j = 0; j < dataLinesOnAllImages[i].size(); j++) {
			line(image, dataLinesOnAllImages[i][j].hh, dataLinesOnAllImages[i][j].hl, Scalar(rand() % 256, rand() % 256, rand() % 256, 255), dataLinesOnAllImages[i][j].median);
		}
		struct color {
			size_t r, g, b;
			void operator+=(RGB tmp) {
				r += tmp.r;
				g += tmp.g;
				b += tmp.b;
			}
			bool operator<(RGB tmp) {
				if (r < tmp.r&&g < tmp.g&&b < tmp.b)
					return true;
				return false;
			}
		};

		color* middle = (color*)calloc(dataLinesOnAllImages[i].size(), sizeof(color));
		size_t* NumberMiddle = (size_t*)calloc(dataLinesOnAllImages[i].size(), sizeof(size_t));
		long long whatIsClass = -1;
		std::vector<std::vector<float_t>> resultCALC(dataLinesOnAllImages[i].size());
		for (size_t k = 0; k < resultCALC.size(); k++) {
			std::vector<float_t> tmp(allimage[i].size().height, 0);
			resultCALC[k] = tmp;
		}
		RGB Black = { 0, 0, 0 };
		RGB LastColor = { 0,0,0 };
		for (size_t l = 0; l < image.size().height; l++) {
			for (size_t j = 0; j < image.size().width; j++) {
				if (ptr[l* image.size().width + j] != Black) {
					if (LastColor != ptr[l* image.size().width + j]) {
						LastColor = ptr[l* image.size().width + j];
						whatIsClass++;
						if (whatIsClass == dataLinesOnAllImages[i].size())
							whatIsClass = 0;
					}
					middle[whatIsClass] += ptr_im[l* image.size().width + j];
					NumberMiddle[whatIsClass] += 1;
					ptr[l* image.size().width + j] = ptr_im[l* image.size().width + j];
				}
			}
		}
		for (size_t j = 0; j < dataLinesOnAllImages[i].size(); j++) {
			middle[j].r /= NumberMiddle[j];
			middle[j].g /= NumberMiddle[j];
			middle[j].b /= NumberMiddle[j];
		}
		for (size_t l = 0; l < image.size().height; l++) {
			for (size_t j = 0; j < image.size().width; j++) {
				if (ptr[l* image.size().width + j] != Black) {
					if (LastColor != ptr[l * image.size().width + j]) {
						LastColor = ptr[l * image.size().width + j];
						whatIsClass++;
						if (whatIsClass == dataLinesOnAllImages[i].size())
							whatIsClass = 0;
					}
					if (middle[whatIsClass] < ptr_im[l* image.size().width + j]) {
						resultCALC[whatIsClass][l] += 1.0f;
					}
				}
			}
		}
		if (dataLinesOnAllImages[i].size() > 0) {
			size_t step_x = std::min(image.size().width, image.size().height) / (dataLinesOnAllImages[i].size() + 1);
			size_t sizeNumers = step_x / 64;
			for (size_t j = 0; j < dataLinesOnAllImages[i].size(); j++) {
				for (size_t u = 0; u < dataLinesOnAllImages[i].size(); u++) {
					/*if (u == j) */ {
						double result = 0;
						size_t numberNotNULL = 0;
						for (size_t l = 0; l < image.size().height; l++) {
							if (resultCALC[u][l] > 0) {
								if ((resultCALC[j][l] / dataLinesOnAllImages[i][j].median) < (resultCALC[u][l] / dataLinesOnAllImages[i][u].median))
									result += abs((resultCALC[j][l] / dataLinesOnAllImages[i][j].median) / (resultCALC[u][l] / dataLinesOnAllImages[i][u].median));
								else
									result += abs((resultCALC[u][l] / dataLinesOnAllImages[i][u].median) / (resultCALC[j][l] / dataLinesOnAllImages[i][j].median));
								numberNotNULL++;
							}
						}
						result /= numberNotNULL;
						char str[30];
						sprintf(str, "%.4g", (result*100.0f));
						putText(image, str, Point2f(step_x + j * step_x, step_x + u * step_x), FONT_HERSHEY_PLAIN, sizeNumers, Scalar(255, 255, 255, 255), 1, CV_AA);
					}
				}
			}
		}

		allimage[i] = image;
		free(middle);
		free(NumberMiddle);
	}
	BITMAP bitmapInfo;
	const HDC hdcwindow = GetDC(hMainWnd);
	for (int i = 0; i < images_length; i++)
	{
		images[i].hBitmap = (HBITMAP)ConvertCVMatToBMP(allimage[i]);
		GetObject(images[i].hBitmap, sizeof(bitmapInfo), &bitmapInfo);
		images[i].height = bitmapInfo.bmHeight;
		images[i].width = bitmapInfo.bmWidth;
	}
	HDC hdc = CreateCompatibleDC(hdcwindow);
	ReleaseDC(nullptr, hdc);
	InvalidateRect(hMainWnd, NULL, FALSE);
	ExitThread(0);
}

DWORD WINAPI DeletePartSpectrum(LPVOID lpParam) {

#pragma  omp parallel for
	for (int j = 0; j < allimage.size(); j++) {
		const uint width_with_channels = allimage[j].size().width*allimage[j].channels();
		const uint width_image = allimage[j].size().width;
		uint height_image = allimage[j].size().height;
		RGB* dataStart = (RGB*)allimage[j].datastart;
		size_t center_x = 0;
		size_t center_y = 0;
		size_t length = 0;
		size_t middle = 0;
		for (size_t h = 0; h < height_image; h++) {
			for (size_t w = 0; w < width_image; w++) {
				middle += dataStart[h*width_image + w].b;
			}
		}
		middle /= height_image * width_image;
		for (size_t h = 0; h < height_image; h++) {
			for (size_t w = 0; w < width_image; w++) {
				if (dataStart[h*width_image + w].b > middle) {
					dataStart[h*width_image + w].b = 0;
					dataStart[h*width_image + w].g = 0;
					dataStart[h*width_image + w].r = 0;
				}
			}
		}


	}

	HDC hdc = *((HDC*)lpParam);
	BITMAP bitmapInfo;
	const HDC hdcwindow = GetDC(hMainWnd);
	for (int i = 0; i < images_length; i++)
	{
		images[i].hBitmap = (HBITMAP)ConvertCVMatToBMP(allimage[i]);
		GetObject(images[i].hBitmap, sizeof(bitmapInfo), &bitmapInfo);
		images[i].height = bitmapInfo.bmHeight;
		images[i].width = bitmapInfo.bmWidth;
	}
	hdc = CreateCompatibleDC(hdcwindow);
	ReleaseDC(nullptr, hdc);
	InvalidateRect(hMainWnd, NULL, FALSE);

	ExitThread(0);
}
DWORD WINAPI Contrast(LPVOID lpParam) {
#pragma omp parallel for
	for (int j = 0; j < allimage.size(); j++) {
		float kernel[9];
		kernel[0] = -0.1;
		kernel[1] = -0.1;
		kernel[2] = -0.1;

		kernel[3] = -0.1;
		kernel[4] = 2;
		kernel[5] = -0.1;

		kernel[6] = -0.1;
		kernel[7] = -0.1;
		kernel[8] = -0.1;

		Mat kernel_matrix = Mat(3, 3, CV_32FC1, kernel);

		filter2D(allimage[j], allimage[j], allimage[j].type(), kernel_matrix, Point(-1, -1));
		filter2D(allimage[j], allimage[j], allimage[j].type(), kernel_matrix, Point(-1, -1));
		filter2D(allimage[j], allimage[j], allimage[j].type(), kernel_matrix, Point(-1, -1));
	}
	HDC hdc = *((HDC*)lpParam);
	BITMAP bitmapInfo;
	const HDC hdcwindow = GetDC(hMainWnd);
	for (int i = 0; i < images_length; i++)
	{
		images[i].hBitmap = (HBITMAP)ConvertCVMatToBMP(allimage[i]);
		GetObject(images[i].hBitmap, sizeof(bitmapInfo), &bitmapInfo);
		images[i].height = bitmapInfo.bmHeight;
		images[i].width = bitmapInfo.bmWidth;
	}
	hdc = CreateCompatibleDC(hdcwindow);
	ReleaseDC(nullptr, hdc);
	InvalidateRect(hMainWnd, NULL, FALSE);

	ExitThread(0);
}
DWORD WINAPI  FillArea(LPVOID lpParam) {
	struct RGB {
		uchar r, g, b;
		bool operator ==(RGB s) {
			if (s.r == r && s.g == g && s.b == b)
				return true;
			else
				return false;
		}
	};
#pragma omp parallel for
	for (int i = 0; i < allimage.size(); i++) {
		uint width = allimage[i].size().width;
		uint height = allimage[i].size().height;
		POINT pt, tmp;
		std::stack<POINT> spt;
		pt.x = 0;
		pt.y = 0;
		spt.push(pt);
		RGB* src = (RGB*)allimage[i].data;
		RGB mainColor = *src;
		RGB color = { mainColor.r ^ 0xff - 1,mainColor.g ^ 0xff + 1,mainColor.b ^ 0xff - 1 };
		while (!spt.empty()) {
			tmp = spt.top();
			spt.pop();
			src[tmp.y*width + tmp.x] = color;

			if ((tmp.x + 1) < width) {
				if (src[tmp.y*width + tmp.x + 1] == mainColor) {
					pt.x = tmp.x + 1;
					pt.y = tmp.y;
					spt.push(pt);
				}
			}
			if ((tmp.x - 1) > -1) {
				if (src[tmp.y*width + tmp.x - 1] == mainColor) {
					pt.x = tmp.x - 1;
					pt.y = tmp.y;
					spt.push(pt);
				}
			}
			if ((tmp.y - 1) > -1) {
				if (src[(tmp.y - 1)*width + tmp.x] == mainColor) {
					pt.x = tmp.x;
					pt.y = tmp.y - 1;
					spt.push(pt);
				}
			}
			if ((tmp.y + 1) < height) {
				if (src[(tmp.y + 1)*width + tmp.x] == mainColor) {
					pt.x = tmp.x;
					pt.y = tmp.y + 1;
					spt.push(pt);
				}
			}
		}
		RGB color_1 = { 0,0,0 };
		RGB color_2 = { color.r ^ 0xff + 1,color.g ^ 0xff - 1,color.b ^ 0xff + 1 };
		for (int i = 0; i < width*height; i++) {
			if (src[i] == mainColor)
				src[i] = color_2;
		}
		for (int i = 0; i < width*height; i++) {
			if (src[i] == color)
				src[i] = color_1;
		}
	}
	HDC hdc = *((HDC*)lpParam);
	BITMAP bitmapInfo;
	const HDC hdcwindow = GetDC(hMainWnd);
	for (int i = 0; i < images_length; i++)
	{
		images[i].hBitmap = (HBITMAP)ConvertCVMatToBMP(allimage[i]);
		GetObject(images[i].hBitmap, sizeof(bitmapInfo), &bitmapInfo);
		images[i].height = bitmapInfo.bmHeight;
		images[i].width = bitmapInfo.bmWidth;
	}
	hdc = CreateCompatibleDC(hdcwindow);
	ReleaseDC(nullptr, hdc);
	InvalidateRect(hMainWnd, NULL, FALSE);

	ExitThread(0);
}
DWORD WINAPI MakeDraw(LPVOID lpParam) {
	struct RGB {
		uchar r, g, b;
		bool operator ==(RGB s) {
			if (s.r == r && s.g == g && s.b == b)
				return true;
			else
				return false;
		}
		bool operator !=(RGB s) {
			if (s.r != r && s.g != g && s.b != b)
				return true;
			else
				return false;
		}
	};
	dataLinesOnAllImages.resize(allimage.size());
#pragma omp parallel for
	for (int j = 0; j < allimage.size(); j++) {
		const uint width_with_channels = allimage[j].size().width*allimage[j].channels();
		const uint width_image = allimage[j].size().width;
		const uint height_image = allimage[j].size().height;
		RGB* dataStart = (RGB*)allimage[j].datastart;
		RGB Black = { 0,0,0 };
		RGB BLACK_TMP = { 10,10,10 };
		RGB Blue = { 255,0,0 };
		std::vector<std::vector<size_t>> positions(height_image);
		std::vector<std::vector<size_t>> length(height_image);
		for (size_t h = 0; h < height_image; h++) {
			std::vector<size_t> part_positin;
			std::vector<size_t> part_positin_black_line;
			std::vector<size_t> part_length;
			bool find = true;
			for (size_t w = 0; w < width_image; w++) {
				if (dataStart[h*width_image + w] != Black) {
					if (find == true) {
						part_positin.push_back(w);
						find = false;
					}
				}
				else {
					if (!find) {
						part_positin_black_line.push_back(w);
						part_length.push_back(w - part_positin.back());
					}
					find = true;
				}
			}
			positions[h] = part_positin;
			length[h] = part_length;
		}
		for (size_t h = 0; h < height_image; h++) {
			for (size_t j = 0; j < length[h].size(); j++) {
				for (size_t k = j; k < length[h].size(); k++) {
					if (length[h][k] < length[h][j]) {
						size_t tmp = length[h][k];
						length[h][k] = length[h][j];
						length[h][j] = tmp;
					}
				}
			}
		}
		std::vector<size_t> median;
		for (size_t h = 0; h < height_image; h++) {
			if (length[h].size() > 1)
				median.push_back(length[h][length[h].size() / 2]);
		}
		std::sort(median.begin(), median.end());
		int middle = median[median.size() / 2];
		int middle_h = middle * 2;
		size_t height_h = 0;
		size_t height_l = 0;
		for (size_t i = 0; i < height_image; i++) {
			if (dataStart[i*width_image] == Blue && height_h == 0) {
				height_h = i;
			}
			else if (dataStart[i*width_image] == Blue && height_h > 0) {
				height_l = i;
			}
		}
		size_t numbers_next,
			numbers_previous = 0;
		std::vector<Point> hhpositions;
		std::vector<Point> hlpositions;
		for (int h = height_h; h <= height_l - middle_h; h = height_l - middle_h) {
			for (int w = 0; w < width_image - middle; w++) {
				numbers_next = 0;
				for (int hh = h; hh < h + middle_h; hh++) {
					for (int ww = w; ww < w + middle; ww++) {
						if (dataStart[hh*width_image + ww] != Black && dataStart[hh*width_image + ww] != BLACK_TMP) {
							numbers_next++;
						}
					}
				}
				if (numbers_next < numbers_previous) {
					if (numbers_previous > (middle*middle) / 8) {
						if (h == height_h) {
							Point write = { w,h };
							hhpositions.push_back(write);
							write = { w + middle , h };
							hhpositions.push_back(write);
						}
						else if (height_l - middle_h) {
							Point write = { w,h + middle_h };
							hlpositions.push_back(write);
							write = { w + middle ,h + middle_h };
							hlpositions.push_back(write);
						}
						for (size_t hh = h; hh < h + middle_h; hh++) {
							for (size_t ww = w; ww < w + middle; ww++) {
								dataStart[hh*width_image + ww] = BLACK_TMP;
							}
						}
					}
				}
				numbers_previous = numbers_next;
			}
			h++;
			if (h > height_l - middle_h)
				break;
		}
		for (size_t i = 0; i < hlpositions.size(); i += 2) {
			hlpositions.erase(hlpositions.begin() + i);
			hlpositions.erase(hlpositions.begin() + i);
		}
		for (size_t i = 0; i < hhpositions.size(); i += 2) {
			hhpositions.erase(hhpositions.begin() + i);
			hhpositions.erase(hhpositions.begin() + i);
		}
		std::vector<size_t> pointUsedhh;
		std::vector<size_t> pointUsedhl;
		std::vector<allPositions> lines;
		if (!hhpositions.empty() && !hlpositions.empty())
			for (size_t i = 0; i < hhpositions.size(); i += 2) {
				for (size_t k = 0; k < hlpositions.size(); k += 2) {
					if (abs(hhpositions[i].x - hlpositions[k].x) < (middle - 1)) {
						bool find = false;
						for (auto iter : pointUsedhl) {
							if (iter == k)
								find = true;
						}
						for (auto iter : pointUsedhh) {
							if (iter == i)
								find = true;
						}
						if (!find) {
							line(allimage[j], (hhpositions[i + 1] + hhpositions[i]) / 2, (hlpositions[k + 1] + hlpositions[k]) / 2, Scalar(rand() % 256, rand() % 256, rand() % 256, 255), middle, CV_AA);
							allPositions pt;
							pt.hh = (hhpositions[i + 1] + hhpositions[i]) / 2;
							pt.hl = (hlpositions[k + 1] + hlpositions[k]) / 2;
							pt.median = middle;
							lines.push_back(pt);
							pointUsedhl.push_back(k);
							pointUsedhh.push_back(i);
						}
					}
				}
			}
		dataLinesOnAllImages[j] = (lines);
	}
	HDC hdc = *((HDC*)lpParam);
	BITMAP bitmapInfo;
	const HDC hdcwindow = GetDC(hMainWnd);
	for (int i = 0; i < images_length; i++)
	{
		images[i].hBitmap = (HBITMAP)ConvertCVMatToBMP(allimage[i]);
		GetObject(images[i].hBitmap, sizeof(bitmapInfo), &bitmapInfo);
		images[i].height = bitmapInfo.bmHeight;
		images[i].width = bitmapInfo.bmWidth;
	}
	hdc = CreateCompatibleDC(hdcwindow);
	ReleaseDC(nullptr, hdc);
	InvalidateRect(hMainWnd, NULL, FALSE);

	ExitThread(0);
}
DWORD WINAPI filterNoise(LPVOID lpParam)
{
#pragma omp parallel for
	for (int j = 0; j < allimage.size(); j++) {
		const uint width_with_channels = allimage[j].size().width*allimage[j].channels();
		const uint width_image = allimage[j].size().width;
		const uint height_image = allimage[j].size().height;
		RGB* dataStart = (RGB*)allimage[j].datastart;
		RGB colorBLACK = { 0,0,0 };
		RGB blue = { 255,0,0 };
		std::vector<std::vector<size_t>> length(height_image);
		std::vector<std::vector<size_t>> length_shadow(height_image);
		for (size_t h = 0; h < height_image; h++) {
			std::vector<size_t> part_positin;
			std::vector<size_t> part_length;
			std::vector<size_t> part_length_shadow;
			bool find = true;
			for (size_t w = 0; w < width_image; w++) {
				if (dataStart[h*width_image + w] != colorBLACK) {
					if (find == true) {
						if (!part_positin.empty())
							part_length.push_back(w - part_positin.back());
						find = false;
					}
				}
				else {
					if (!find) {
						part_positin.push_back(w);
					}
					find = true;
				}
			}
			length[h] = part_length;
		}
		for (size_t h = 0; h < height_image; h++) {
			for (size_t j = 0; j < length[h].size(); j++) {
				for (size_t k = j; k < length[h].size(); k++) {
					if (length[h][k] < length[h][j]) {
						size_t tmp = length[h][k];
						length[h][k] = length[h][j];
						length[h][j] = tmp;
					}
				}
			}
		}
		std::vector<size_t> median(height_image);
		for (size_t h = 0; h < height_image; h++)
			if (length[h].size() > 0)
				median[h] = length[h][length[h].size() / 2];
		size_t hhLine = 0;
		size_t hlLine = 0;

		for (size_t h = 0; h < height_image; h++) {
			if (hhLine == 0)
				if (dataStart[h*width_image] == blue)
					hhLine = h;
			if (hhLine != 0)
				if (dataStart[h*width_image] == blue)
					hlLine = h;
		}
		hhLine++;
		hlLine--;
		if (hhLine == 1 && hlLine == 18446744073709551615) {
			hlLine = height_image - 1;
		}
		size_t line_size = hlLine - hhLine;
		for (size_t w = 0; w < width_image - 3; w += 3) {
			size_t number = 0;
			for (size_t i = w; i < w + 3; i++)
				for (size_t h = hhLine; h < hlLine; h++) {
					if (dataStart[h*width_image + i] != colorBLACK)
						number++;
				}
			if (number < (line_size / 2)) {
				for (size_t i = w; i < w + 3; i++)
					for (size_t h = hhLine; h < hlLine; h++)
						dataStart[h*width_image + i] = colorBLACK;
			}
		}
	}
	HDC hdc = *((HDC*)lpParam);
	BITMAP bitmapInfo;
	const HDC hdcwindow = GetDC(hMainWnd);
	for (int i = 0; i < images_length; i++)
	{
		images[i].hBitmap = (HBITMAP)ConvertCVMatToBMP(allimage[i]);
		GetObject(images[i].hBitmap, sizeof(bitmapInfo), &bitmapInfo);
		images[i].height = bitmapInfo.bmHeight;
		images[i].width = bitmapInfo.bmWidth;
	}
	hdc = CreateCompatibleDC(hdcwindow);
	ReleaseDC(nullptr, hdc);
	InvalidateRect(hMainWnd, NULL, FALSE);

	ExitThread(0);
}
DWORD WINAPI findCenter(LPVOID lpParam) {
	struct RGB {
		uchar r, g, b;
		bool operator ==(RGB s) {
			if (s.r == r && s.g == g && s.b == b)
				return true;
			else
				return false;
		}
		bool operator !=(RGB s) {
			if (s.r != r && s.g != g && s.b != b)
				return true;
			else
				return false;
		}
	};
#pragma omp parallel for
	for (int j = 0; j < allimage.size(); j++) {
		const uint width_with_channels = allimage[j].size().width*allimage[j].channels();
		const uint width_image = allimage[j].size().width;
		uint height_image = allimage[j].size().height;
		RGB* dataStart = (RGB*)allimage[j].datastart;
		size_t center_x = 0;
		size_t center_y = 0;
		size_t length = 0;
		RGB color = { 0,0,0 };
		for (size_t h = 0; h < height_image; h++) {
			for (size_t w = 0; w < width_image; w++) {
				if (dataStart[h*width_image + w] != color) {
					center_x += w;
					center_y += h;
					length++;
				}
			}
		}

		size_t size = 100;
		size_t size_x = 200;
		size_t size_y = 100;
		size_t head = 1, tail = 0;
		while (head != tail) {
			if (length > 0) {
				tail = head;
				head = 0;
				size_y += 5;
				size_t _center_x = center_x / length;
				size_t _center_y = center_y / length;
				_center_x -= size_x;
				_center_y -= size_y;
				if (_center_x < 0)
					_center_x = 0;
				if (_center_y < 0)
					_center_y = 0;
				uint height_start = _center_y + size_y * 2;
				uint width_start = _center_x + size_x * 2;
				if (height_start > height_image)
					height_start = height_image;
				if (width_start > width_image)
					width_start = width_image;
				for (size_t h = _center_y; h < height_start; h++) {
					for (size_t w = _center_x; w < width_start; w++) {
						if (dataStart[h*width_image + w] != color) {
							head++;
						}
						//dataStart[h*width_image + w].r = 255;
						//dataStart[h*width_image + w].b = 255;
					}
				}
			}
		}
		size_t _center_y = center_y / length;
		size_t hh_center_y = _center_y - size_y;
		size_t hl_center_y = _center_y + size_y;
		for (size_t i = 0; i < height_image; i++)
		{
			if (hh_center_y > i || hl_center_y < i)
				for (size_t j = 0; j < width_image; j++) {
					dataStart[i*width_image + j] = color;
				}
		}
		for (size_t j = 0; j < width_image; j++) {
			dataStart[hh_center_y*width_image + j] = { 255,0,0 };
		}
		for (size_t j = 0; j < width_image; j++) {
			dataStart[hl_center_y*width_image + j] = { 255,0,0 };
		}
	}
	HDC hdc = *((HDC*)lpParam);
	BITMAP bitmapInfo;
	const HDC hdcwindow = GetDC(hMainWnd);
	for (int i = 0; i < images_length; i++)
	{
		images[i].hBitmap = (HBITMAP)ConvertCVMatToBMP(allimage[i]);
		GetObject(images[i].hBitmap, sizeof(bitmapInfo), &bitmapInfo);
		images[i].height = bitmapInfo.bmHeight;
		images[i].width = bitmapInfo.bmWidth;
	}
	hdc = CreateCompatibleDC(hdcwindow);
	ReleaseDC(nullptr, hdc);
	InvalidateRect(hMainWnd, NULL, FALSE);

	ExitThread(0);
}

DWORD WINAPI findLine(LPVOID lpParam)
{
#pragma omp parallel for
	for (int j = 0; j < allimage.size(); j++) {
		const uint width_with_channels = allimage[j].size().width*allimage[j].channels();
		const uint width_image = allimage[j].size().width;
		const uint height_image = allimage[j].size().height;
		RGB* dataStart = (RGB*)allimage[j].datastart;
		std::vector<int> length(height_image);
		for (size_t h = 0; h < height_image; h++) {
			for (size_t w = 0; w < width_image; w++) {
				if (dataStart[h*width_image + w].b != 0 && dataStart[h*width_image + w].g != 0 && dataStart[h*width_image + w].r != 0) {
					length[h] += 1;
				}
			}
		}

		int max = 0;
		for (size_t h = 0; h < height_image; h++) {
			if (max < length[h])
				max = length[h];
		}
		for (size_t h = 0; h < height_image; h++) {
			if (length[h] < max / 2) {
				for (size_t j = 0; j < width_image; j++) {
					dataStart[h*width_image + j] = { 0,0,0 };
				}
			}

		}
	}
	HDC hdc = *((HDC*)lpParam);
	BITMAP bitmapInfo;
	const HDC hdcwindow = GetDC(hMainWnd);
	for (int i = 0; i < images_length; i++)
	{
		images[i].hBitmap = (HBITMAP)ConvertCVMatToBMP(allimage[i]);
		GetObject(images[i].hBitmap, sizeof(bitmapInfo), &bitmapInfo);
		images[i].height = bitmapInfo.bmHeight;
		images[i].width = bitmapInfo.bmWidth;
	}
	hdc = CreateCompatibleDC(hdcwindow);
	ReleaseDC(nullptr, hdc);
	InvalidateRect(hMainWnd, NULL, FALSE);

	ExitThread(0);
}
DWORD WINAPI MedianFilter(LPVOID lpParam)
{
#pragma omp parallel for
	for (int j = 0; j < allimage.size(); j++) {
		medianBlur(allimage[j], allimage[j], 13);

	}

	HDC hdc = *((HDC*)lpParam);
	BITMAP bitmapInfo;
	const HDC hdcwindow = GetDC(hMainWnd);
	for (int i = 0; i < images_length; i++)
	{
		images[i].hBitmap = (HBITMAP)ConvertCVMatToBMP(allimage[i]);
		GetObject(images[i].hBitmap, sizeof(bitmapInfo), &bitmapInfo);
		images[i].height = bitmapInfo.bmHeight;
		images[i].width = bitmapInfo.bmWidth;
	}
	hdc = CreateCompatibleDC(hdcwindow);
	ReleaseDC(nullptr, hdc);
	InvalidateRect(hMainWnd, NULL, FALSE);

	ExitThread(0);
}
LRESULT CALLBACK MainWinProc(HWND, UINT, WPARAM, LPARAM);
HINSTANCE hInstance;
ATOM RegMyWindowClass(HINSTANCE hInst, LPCTSTR lpzClassName)
{
	WNDCLASS wcWindowClass = { 0 };
	wcWindowClass.lpfnWndProc = (WNDPROC)MainWinProc;
	wcWindowClass.style = CS_HREDRAW | CS_VREDRAW;
	wcWindowClass.hInstance = hInst;
	wcWindowClass.lpszClassName = lpzClassName;
	wcWindowClass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wcWindowClass.hbrBackground = (HBRUSH)GetStockObject(GRAY_BRUSH);
	return RegisterClass(&wcWindowClass);
}
int WINAPI WinMain(HINSTANCE hInst, HINSTANCE, LPSTR, int ss)
{
	LPCTSTR lpzClass = TEXT("My Window Class!");
	if (!RegMyWindowClass(hInst, lpzClass)) return FALSE;
	hMainWnd = CreateWindow("My Window Class!", "OPENCV", WS_VSCROLL | WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, NULL, NULL, hInst, NULL);
	if (!hMainWnd) return FALSE;
	hInstance = hInst;
	ShowWindow(hMainWnd, ss);
	UpdateWindow(hMainWnd);
	MSG msg;
	while (GetMessage(&msg, (HWND)NULL, 0, 0)) {
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}
	return msg.wParam;
}

SCROLLINFO si;
LRESULT CALLBACK MainWinProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
	switch (msg) {
		PAINTSTRUCT ps;
	case WM_SIZE:
	{
		RECT rc = { 0 };
		GetClientRect(hwnd, &rc);

		size_t length = (rc.right - rc.left) / (images[0].width);
		if (length == 0)
			length = 1;
		si.cbSize = sizeof(SCROLLINFO);
		si.fMask = SIF_ALL;
		si.nMin = 0;
		si.nMax = (images_length / length)* (images[0].height + 5) + 50;
		si.nPage = (rc.bottom - rc.top);
		si.nPos = 0;
		si.nTrackPos = 0;
		SetScrollInfo(hwnd, SB_VERT, &si, true);
		return 0;
	}
	case WM_CREATE:
	{
		{
			OPENFILENAME OpenFile;
			CHAR file[2048] = { 0 };
			CHAR readWay[2048];
			memset(&OpenFile, 0, sizeof(OpenFile));
			OpenFile.lStructSize = sizeof(OPENFILENAME);
			OpenFile.hwndOwner = GetActiveWindow();
			OpenFile.lpstrFile = (LPSTR)file;
			OpenFile.nMaxFile = sizeof(file);
			OpenFile.lpstrFilter = "JPEG files(*.jpg)\0*.jpg\0JPEG files(*.jpg)\0*.jpg\0All files(*.*)\0*.*\0\0";
			OpenFile.nFilterIndex = 1;
			OpenFile.lpstrTitle = "Поиск картинки";
			OpenFile.lpstrInitialDir = NULL;
			OpenFile.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_HIDEREADONLY | OFN_ALLOWMULTISELECT;
			if (GetOpenFileName(&OpenFile) == TRUE)
			{
				size_t index = 0;
				strcpy(readWay, OpenFile.lpstrFile);
				CHAR* writeName = readWay;
				while (*writeName != 0)
					writeName++;
				OpenFile.lpstrFile += OpenFile.nFileOffset;
				while ((*OpenFile.lpstrFile != 0) && (OpenFile.nMaxFile != 0))
				{
					strcat(readWay, "\\");
					strcat(readWay, OpenFile.lpstrFile);
					namesReads.resize(namesReads.size() + 1);
					for (size_t i = 0; readWay[i] != 0; i++)
						namesReads[namesReads.size() - 1].push_back(readWay[i]);
					allimage.push_back(imread(readWay));
					CHAR* deleteName = writeName;
					while (*deleteName != 0)
						*deleteName = 0,
						deleteName++;
					while (*(OpenFile.lpstrFile++) && (--OpenFile.nMaxFile));
				}
			}
		}
		const char* names[] = { "MedianFilter",
											"Contrast",
											"DeletePartSpectrum",
											"DeleteMinimumData",
											"findLine",
											"findCenter",
											"fillArea",
											"MakeDraw",
											"WriteImages",
											"Last step"
		};
		hfontMSSS = CreateFont(1, 0, FW_BLACK, FW_BLACK, FW_BLACK, FALSE, FALSE, FALSE, ANSI_CHARSET, OUT_TT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, FF_DECORATIVE, "MS Sans Serif");
		for (size_t i = 0; i < sizeof(names) / sizeof(char*); i++) {
			hButton[i] = CreateWindow("button", TEXT(names[i]), WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, 10 + i * 100, 10, 100, 30, hwnd, (HMENU)(15000 + i), hInstance, NULL);
			SendMessage(hButton[i], WM_SETFONT, reinterpret_cast<WPARAM>(hfontMSSS), MAKELPARAM(1, 0));
		}

		const HDC hdcwindow = GetDC(hwnd);
		images_length = allimage.size();
		images = new Image[images_length];
		BITMAP bitmapInfo;

		for (int i = 0; i < images_length; i++)
		{
			images[i].hBitmap = (HBITMAP)ConvertCVMatToBMP(allimage[i]);
			GetObject(images[i].hBitmap, sizeof(bitmapInfo), &bitmapInfo);
			images[i].height = bitmapInfo.bmHeight;
			images[i].width = bitmapInfo.bmWidth;
		}
		imageDC = CreateCompatibleDC(hdcwindow);
		ReleaseDC(nullptr, imageDC);
		return 0;
	}

	case WM_COMMAND:
		switch (LOWORD(wp)) {
		case 15000:
		{
			HANDLE WorkerThread = CreateThread(NULL, 0, &MedianFilter, (PVOID)&imageDC, 0, NULL);
			for (int i = 0; i < sizeof(hButton) / sizeof(HWND); i++)
				EnableWindow(hButton[i], FALSE);
			DWORD dwWaitResult = ::WaitForSingleObject(
				WorkerThread, INFINITE);
			for (int i = 0; i < sizeof(hButton) / sizeof(HWND); i++)
				EnableWindow(hButton[i], TRUE);
			CloseHandle(WorkerThread);
			return 0;
		}
		case 15001:
		{
			HANDLE WorkerThread = CreateThread(NULL, 0, &Contrast, (PVOID)&imageDC, 0, NULL);
			for (int i = 0; i < sizeof(hButton) / sizeof(HWND); i++)
				EnableWindow(hButton[i], FALSE);
			DWORD dwWaitResult = ::WaitForSingleObject(
				WorkerThread, INFINITE);
			for (int i = 0; i < sizeof(hButton) / sizeof(HWND); i++)
				EnableWindow(hButton[i], TRUE);
			CloseHandle(WorkerThread);
			InvalidateRect(hMainWnd, NULL, FALSE);
			return 0;
		}
		case 15002:
		{
			HANDLE WorkerThread = CreateThread(NULL, 0, &DeletePartSpectrum, (PVOID)&imageDC, 0, NULL);
			for (int i = 0; i < sizeof(hButton) / sizeof(HWND); i++)
				EnableWindow(hButton[i], FALSE);
			DWORD dwWaitResult = ::WaitForSingleObject(
				WorkerThread, INFINITE);
			for (int i = 0; i < sizeof(hButton) / sizeof(HWND); i++)
				EnableWindow(hButton[i], TRUE);
			CloseHandle(WorkerThread);
			InvalidateRect(hMainWnd, NULL, FALSE);
			return 0;
		}
		case 15003:
		{
			HANDLE WorkerThread = CreateThread(NULL, 0, &filterNoise, (PVOID)&imageDC, 0, NULL);
			for (int i = 0; i < sizeof(hButton) / sizeof(HWND); i++)
				EnableWindow(hButton[i], FALSE);
			DWORD dwWaitResult = ::WaitForSingleObject(
				WorkerThread, INFINITE);
			for (int i = 0; i < sizeof(hButton) / sizeof(HWND); i++)
				EnableWindow(hButton[i], TRUE);
			CloseHandle(WorkerThread);
			InvalidateRect(hMainWnd, NULL, FALSE);
			return 0;
		}
		case 15004:
		{
			HANDLE WorkerThread = CreateThread(NULL, 0, &findLine, (PVOID)&imageDC, 0, NULL);
			for (int i = 0; i < sizeof(hButton) / sizeof(HWND); i++)
				EnableWindow(hButton[i], FALSE);
			DWORD dwWaitResult = ::WaitForSingleObject(
				WorkerThread, INFINITE);
			for (int i = 0; i < sizeof(hButton) / sizeof(HWND); i++)
				EnableWindow(hButton[i], TRUE);
			CloseHandle(WorkerThread);
			InvalidateRect(hMainWnd, NULL, FALSE);
			return 0;
			
		}
		case 15005:
		{
			HANDLE WorkerThread = CreateThread(NULL, 0, &findCenter, (PVOID)&imageDC, 0, NULL);
			for (int i = 0; i < sizeof(hButton) / sizeof(HWND); i++)
				EnableWindow(hButton[i], FALSE);
			DWORD dwWaitResult = ::WaitForSingleObject(
				WorkerThread, INFINITE);
			for (int i = 0; i < sizeof(hButton) / sizeof(HWND); i++)
				EnableWindow(hButton[i], TRUE);
			CloseHandle(WorkerThread);
			InvalidateRect(hMainWnd, NULL, FALSE);
			return 0;
		}
		case 15006:
		{
			HANDLE WorkerThread = CreateThread(NULL, 0, &FillArea, (PVOID)&imageDC, 0, NULL);
			for (int i = 0; i < sizeof(hButton) / sizeof(HWND); i++)
				EnableWindow(hButton[i], FALSE);
			DWORD dwWaitResult = ::WaitForSingleObject(
				WorkerThread, INFINITE);
			for (int i = 0; i < sizeof(hButton) / sizeof(HWND); i++)
				EnableWindow(hButton[i], TRUE);
			CloseHandle(WorkerThread);
			InvalidateRect(hMainWnd, NULL, FALSE);
			return 0;
		}
		case 15007:
		{
			HANDLE WorkerThread = CreateThread(NULL, 0, &MakeDraw, (PVOID)&imageDC, 0, NULL);
			for (int i = 0; i < sizeof(hButton) / sizeof(HWND); i++)
				EnableWindow(hButton[i], FALSE);
			DWORD dwWaitResult = ::WaitForSingleObject(
				WorkerThread, INFINITE);
			for (int i = 0; i < sizeof(hButton) / sizeof(HWND); i++)
				EnableWindow(hButton[i], TRUE);
			CloseHandle(WorkerThread);
			InvalidateRect(hMainWnd, NULL, FALSE);
			return 0;
		}
		case 15008:
		{
			for (size_t i = 0; i < allimage.size(); i++) {
				std::stringstream ss;
				ss << i;
				ss << ".jpg";
				imwrite(ss.str(), allimage[i]);
			}
			return 0;
		}
		case 15009:
		{
			HANDLE WorkerThread = CreateThread(NULL, 0, &CalculateCompare, (PVOID)&imageDC, 0, NULL);
			for (int i = 0; i < sizeof(hButton) / sizeof(HWND); i++)
				EnableWindow(hButton[i], FALSE);
			DWORD dwWaitResult = ::WaitForSingleObject(
				WorkerThread, INFINITE);
			for (int i = 0; i < sizeof(hButton) / sizeof(HWND); i++)
				EnableWindow(hButton[i], TRUE);
			CloseHandle(WorkerThread);
			InvalidateRect(hMainWnd, NULL, FALSE);
			return 0;
		}
		};
	case WM_PAINT:
	{
		if (hwnd)
		{
			RECT wh;
			GetWindowRect(hwnd, &wh);
			int width = wh.right - wh.left;
			const HDC hdc = BeginPaint(hwnd, &ps);
			int X = -GetScrollPos(hwnd, SB_HORZ);
			int Y = -GetScrollPos(hwnd, SB_VERT);
			int r = SaveDC(hdc);
			FillRect(hdc, &ps.rcPaint, GetSysColorBrush(COLOR_3DFACE));
			RestoreDC(hdc, r);
			int x = X;
			int y = Y + 50;
			for (int i = 0; i < images_length; i++)
			{
				SelectObject(imageDC, images[i].hBitmap);
				BitBlt(
					hdc,
					x, y,
					(int)images[i].width,
					(int)images[i].height,
					imageDC,
					0, 0,
					SRCCOPY
				);
				x += images[i].width + 5;
				if (x + images[i].width + 5 > width)
				{
					y += images[i].height + 5;
					x = 0;
				}
			}

			EndPaint(hwnd, &ps);
		}
		return 0;
	}
	case WM_LBUTTONDOWN:
	{
		SCROLLINFO si = { 0 };
		si.cbSize = sizeof(SCROLLINFO);
		si.fMask = SIF_POS;
		si.nPos = 0;
		si.nTrackPos = 0;
		GetScrollInfo(hwnd, SB_VERT, &si);
		break;
	}
	case WM_VSCROLL:
	{
		int nNewPos;
		nNewPos = GetScrollPos(hwnd, msg == WM_HSCROLL ? SB_HORZ : SB_VERT);
		switch (LOWORD(wp))
		{
		case SB_THUMBTRACK:
		case SB_THUMBPOSITION:
			nNewPos = HIWORD(wp);
			break;
		case SB_PAGELEFT:
			nNewPos -= 20;
			break;
		case SB_PAGERIGHT:
			nNewPos += 20;
			break;
		case SB_LINERIGHT:
			nNewPos += 1;
			break;
		case SB_LINELEFT:
			nNewPos -= 1;
			break;
		case SB_ENDSCROLL:
			InvalidateRect(hwnd, NULL, TRUE);
			return TRUE;
		}
		SetScrollPos(hwnd, msg == WM_HSCROLL ? SB_HORZ : SB_VERT, nNewPos, TRUE);
		InvalidateRect(hwnd, NULL, TRUE);
		break;
	}
	case WM_DESTROY:
		PostQuitMessage(0);
		return 0;
	case WM_ERASEBKGND:
		return 0;
	}

	return DefWindowProc(hwnd, msg, wp, lp);
}