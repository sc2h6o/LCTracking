
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include "CompressiveTracker.h"

using namespace cv;
using namespace std;

Rect box; // tracking object
Rect box_test;
bool drawing_box = false;
bool moving_box = false;
bool gotBB = false;	// got tracking box or not
bool fromfile = true;
string video;

string seqName;
string outDir;
string seqDir;
int startFrame = 1;
int endFrame = INT_MAX;


void readBB(const string& file)	// get tracking box from file
{
	ifstream tb_file (file);
	string line;
	getline(tb_file, line);
	istringstream linestream(line);
	string x1, y1, w1, h1;
	getline(linestream, x1, ',');
	getline(linestream, y1, ',');
	if (!y1.empty()) {
		getline(linestream, w1, ',');
		getline(linestream, h1, ',');
	}
	else{
		istringstream linestream(line);
		linestream >> x1 >> y1 >> w1 >> h1;
	}
	int x = atoi(x1.c_str());
	int y = atoi(y1.c_str());
	int w = atoi(w1.c_str());
	int h = atoi(h1.c_str());
	box = Rect(x, y, w, h);
}

// tracking box mouse callback
void mouseHandlerInit(int event, int x, int y, int flags, void *param)
{
	switch (event)
	{
	case CV_EVENT_MOUSEMOVE:
		if (drawing_box)
		{
			box.width = x - box.x;
			box.height = y - box.y;
		}
		break;
	case CV_EVENT_LBUTTONDOWN:
		drawing_box = true;
		box = Rect(x, y, 0, 0);
		break;
	case CV_EVENT_LBUTTONUP:
		drawing_box = false;
		if (box.width < 0)
		{
			box.x += box.width;
			box.width *= -1;
		}
		if( box.height < 0 )
		{
			box.y += box.height;
			box.height *= -1;
		}
		gotBB = true;
		break;
	default:
		break;
	}
}

void mouseHandlerTest(int event, int x, int y, int flags, void *param)
{
	switch (event)
	{
	case CV_EVENT_MOUSEMOVE:
		if (moving_box)
		{
			box_test.x = x - 0.5 * box_test.width;
			box_test.y = y - 0.5 * box_test.height;
		}
		break;
	default:
		break;
	}
}

void print_help(void)
{
	printf("use:\n     welcome to use CompressiveTracking\n");
	printf("Kaihua Zhang's paper:Real-Time Compressive Tracking\n");
	printf("C++ implemented by yang xian\nVersion: 1.0\nEmail: yang_xian521@163.com\nDate:	2012/08/03\n\n");
	printf("-v    source video\n-b        tracking box file\n");
}

void read_options(int argc, char** argv, VideoCapture& capture)
{
	for (int i=0; i<argc; i++)
	{
		if (strcmp(argv[i], "-b") == 0)	// read tracking box from file
		{
			if (argc>i)
			{
				readBB(argv[i+1]);
				gotBB = true;
			}
			else
			{
				print_help();
			}
		}
		if (strcmp(argv[i], "-v") == 0)	// read video from file
		{
			if (argc > i)
			{
				video = string(argv[i+1]);
				capture.open(video);
				fromfile = true;
			}
			else
			{
				print_help();
			}
		}

		if (strcmp(argv[i], "-outdir") == 0)	// read video from file
		{
			if (argc > i)
			{
				outDir = string(argv[i + 1]);
			}
			else
			{
				print_help();
			}
		}

		if (strcmp(argv[i], "-seqname") == 0)	// read video from file
		{
			if (argc > i)
			{
				seqName = string(argv[i + 1]);
			}
			else
			{
				print_help();
			}
		}

		if (strcmp(argv[i], "-seqdir") == 0)	// read video from file
		{
			if (argc > i)
			{
				seqDir = string(argv[i + 1]);
			}
			else
			{
				print_help();
			}
		}
	}
}

int main(int argc, char * argv[])
{
	SeqCapture capture;
	BoxWriter bw;
	//capture.open(0);

	// Read options
	read_options(argc, argv, capture);
	assert(seqName.size() > 0 && seqDir.size() > 0);
	string output = outDir + "/" +seqName + "_lrf.txt";
	string imgdir = seqDir + "/" + seqName + "/" + "img";
	string boxfile = seqDir + "/" + seqName + "/" + "groundtruth_rect.txt";

	if (seqName == "david" || seqName == "David"){
		startFrame = 300;
	}

	capture.setRange(startFrame, endFrame);
	capture.setNameFormat(4, ".jpg");
	capture.open(imgdir, true);
	bw.open(output);
	readBB(boxfile);
	gotBB = true;
	// Init camera
	if (!capture.isOpened())
	{
		cout << "capture device failed to open!" << endl;
		waitKey(0);
		return 1;
	}
	// Register mouse callback to draw the tracking box
	namedWindow("CT", CV_WINDOW_AUTOSIZE);
	setMouseCallback("CT", mouseHandlerInit, NULL);
	// CT framework
	CompressiveTracker ct;

	Mat frame;
	Mat last_gray;
	Mat first;
	if (fromfile)
	{
		capture >> frame;
		cvtColor(frame, last_gray, CV_RGB2GRAY);
		frame.copyTo(first);
	}
	else
	{
		capture.set(CV_CAP_PROP_FRAME_WIDTH, 360);
		capture.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
	}

	// Initialization
	while(!gotBB)
	{
		if (!fromfile)
		{
			capture >> frame;
		}
		else
		{
			first.copyTo(frame);
		}
		cvtColor(frame, last_gray, CV_RGB2GRAY);
		rectangle(frame, box, Scalar(0,0,255));
		imshow("CT", frame);
		if (cvWaitKey(10) == 'q') {	return 0; }
	}
	
	// Remove callback
	setMouseCallback("CT", mouseHandlerTest, NULL);
	printf("Initial Tracking Box = x:%d y:%d h:%d w:%d\n", box.x, box.y, box.width, box.height);
	// CT initialization
	ct.init(last_gray, frame, box);

	// Run-time
	Mat current_gray, display;
	//bw.write(box);
	while(moving_box || capture.read(frame))
	{
		// get frame
		cvtColor(frame, current_gray, CV_RGB2GRAY);
		// Process Frame
		if (!moving_box)
			ct.processFrame(current_gray, frame, box);
		else
			ct.testFrame(current_gray, frame, box_test);
		// Draw Points
		frame.copyTo(display);
		rectangle(display, box, Scalar(0, 0, 255));
		rectangle(display, box_test, Scalar(0, 255, 0));
		// Display
		imshow("prob", current_gray);
		imshow("CT", display);
		bw.write(box);
		
		//printf("Current Tracking Box = x:%d y:%d h:%d w:%d\n", box.x, box.y, box.width, box.height);

		int key = cvWaitKey(1);
		if (key == 'q') { break; }
		if (key == 'p') {
			if (!moving_box) {
				box_test = box;
				moving_box = true;
			} 
			else {
				box_test = Rect(0, 0, 0, 0);
				moving_box = false;
			}

		}
	}
	bw.close();
	return 0;
}