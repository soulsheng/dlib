// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example shows how to use the correlation_tracker from the dlib C++ library.  This
    object lets you track the position of an object as it moves from frame to frame in a
    video sequence.  To use it, you give the correlation_tracker the bounding box of the
    object you want to track in the current video frame.  Then it will identify the
    location of the object in subsequent frames.

    In this particular example, we are going to run on the video sequence that comes with
    dlib, which can be found in the examples/video_frames folder.  This video shows a juice
    box sitting on a table and someone is waving the camera around.  The task is to track the
    position of the juice box as the camera moves around.
*/

#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/dir_nav.h>


//using namespace dlib;
using namespace std;

#define ENABLE_OCV_MP4		1
#define	IMAGE_FILE_MP4		"20160112_dsst.mp4"
#define	IMAGE_FILE_MP4_OUT	"20160112_dsst.avi"

//#if ENABLE_OCV_MP4
#include <opencv.hpp>
#include <cv.h>
//using namespace cv;
#include <string>

void expand_imgs( IplImage* img1, IplImage* img2, IplImage* expanded )  
{  
	cvZero( expanded );  
	cvSetImageROI( expanded, cvRect( 0, 0, img1->width, img1->height ) );  
	cvAdd( img1, expanded, expanded, NULL );  
	cvSetImageROI( expanded, cvRect(img1->width, 0, img2->width, img2->height) );  
	cvAdd( img2, expanded, expanded, NULL );  
	cvResetImageROI( expanded );  
}
void load_image( dlib::array2d<unsigned char>& img, cv::Mat& imageIn)
{
	int cols = imageIn.cols;
	int rows = imageIn.rows;

	img.set_size(rows, cols);
	for ( int i = 0; i < rows; i++ )
	{
		for (int j = 0; j < cols; j++)
		{
			img[i][j] = imageIn.at<uchar>( i, j );
		}
	}
}
void save_image( dlib::array2d<unsigned char>& img, cv::Mat& imageIn)
{
	int cols = imageIn.cols;
	int rows = imageIn.rows;

	imageIn.create(rows, cols, CV_8UC1);
	for ( int i = 0; i < rows; i++ )
	{
		for (int j = 0; j < cols; j++)
		{
			imageIn.at<uchar>(i, j) = img[i][j];
		}
	}
}
//#endif

int main(int argc, char** argv) try
{
#if ENABLE_OCV_MP4
	CvCapture* capture = 0;
	cv::Mat imageIn;
	IplImage iplImgOut;

	capture = cvCaptureFromAVI( IMAGE_FILE_MP4 );

	double fps = cvGetCaptureProperty(capture,CV_CAP_PROP_FPS);   
	CvSize size = cvSize(
		(int)cvGetCaptureProperty( capture, CV_CAP_PROP_FRAME_WIDTH),  
		(int)cvGetCaptureProperty( capture, CV_CAP_PROP_FRAME_HEIGHT));    

	CvSize sizeMix(size);
	sizeMix.width *= 2;

	CvVideoWriter* writer = cvCreateVideoWriter(  
		IMAGE_FILE_MP4_OUT, CV_FOURCC('D', 'I', 'V', 'X'),fps,sizeMix);  

	//DarkChannelGPU	m_DarkChannel;
	//m_DarkChannel.initialize( size.width, size.height );

	if( capture )
	{
		IplImage* iplImg = cvQueryFrame( capture );
		imageIn = iplImg;

		IplImage* iplImgGray = cvCreateImage( 
			cvSize( iplImg->width, iplImg->height ), IPL_DEPTH_8U, 1 );
		cvCvtColor( iplImg, iplImgGray, CV_RGB2GRAY );

		// Load the first frame.  
		cv::Mat imageInGray = iplImgGray;
		dlib::array2d<unsigned char> img;
		load_image(img, imageInGray);
		dlib::correlation_tracker tracker;
		tracker.start_track(img, dlib::centered_rect(dlib::point(839,270), 136, 224));// (dlib::point(839,270), 100, 200))

		IplImage* expanded = cvCreateImage( sizeMix, IPL_DEPTH_8U, 3 );  

		cout << "In capture ..." << endl;
		int nFrameCount = 0;
		int nFrameCountMax = 850;
		for(;nFrameCount<nFrameCountMax;nFrameCount++)
		{
			iplImg = cvQueryFrame( capture );
			imageIn = iplImg;

			cv::Mat imageOut= imageIn.clone();

			//m_DarkChannel.Enhance( imageIn, imageOut );
			cvCvtColor( iplImg, iplImgGray, CV_RGB2GRAY );
			imageInGray = iplImgGray;
			load_image(img, imageInGray);
			tracker.update(img);

			//cv::imshow("imageIn", imageIn );
			//cv::imshow("imageOut", imageOut );

			dlib::drectangle rect = tracker.get_position();
			cv::rectangle(imageOut, cv::Rect( cv::Point(rect.left(), rect.bottom()), cv::Point(rect.right(), rect.top()) ), cv::Scalar(0.0f,255.0f,0.0f) );
#if 0
			iplImgOut = imageOut;
			expand_imgs( iplImg, &iplImgOut, expanded );

			imageOut = expanded;

			cv::imshow("imageOut", imageOut );

			cvWriteToAVI( writer, expanded );  
#else
			cv::imshow("imageOut", imageOut );
			*expanded = imageOut;
			cvWriteToAVI( writer, expanded ); 
#endif
			if( cv::waitKey( 10 ) >= 0 )
				goto _cleanup_;
		}

		cv::waitKey(0);

_cleanup_:
		cvReleaseVideoWriter( &writer ); 
		cvReleaseCapture( &capture );
	}

#else

    if (argc != 2)
    {
        cout << "Call this program like this: " << endl;
        cout << "./video_tracking_ex ../video_frames" << endl;
        return 1;
    }

    // Get the list of video frames.  
    std::vector<dlib::file> files = get_files_in_directory_tree(argv[1], dlib::match_ending(".jpg"));
    std::sort(files.begin(), files.end());
    if (files.size() == 0)
    {
        cout << "No images found in " << argv[1] << endl;
        return 1;
    }

    // Load the first frame.  
    dlib::array2d<unsigned char> img;
    dlib::load_image(img, files[0]);
    // Now create a tracker and start a track on the juice box.  If you look at the first
    // frame you will see that the juice box is centered at pixel point(92,110) and 38
    // pixels wide and 86 pixels tall.
    dlib::correlation_tracker tracker;
    tracker.start_track(img, dlib::centered_rect(dlib::point(93,110), 38, 86));

    // Now run the tracker.  All we have to do is call tracker.update() and it will keep
    // track of the juice box!
    //dlib::image_window win;
    for (unsigned long i = 1; i < files.size(); ++i)
    {
		cv::Mat imageCV = cv::imread( files[i] );
		cv::Mat imageGray = cv::imread( files[i],0 );
		dlib::array2d<unsigned char> img;
#if 0
        load_image(img, files[i]);
#else
		load_image(img, imageGray);
#endif
		tracker.update(img);

        //win.set_image(img); 
        //win.clear_overlay(); 
        //win.add_overlay(tracker.get_position());
		dlib::drectangle rect = tracker.get_position();
		cv::rectangle(imageCV, cv::Rect( cv::Point(rect.left(), rect.bottom()), cv::Point(rect.right(), rect.top()) ), cv::Scalar(0.0f,255.0f,0.0f) );
		cv::imshow("imageCV", imageCV );
		cv::waitKey( 10 );
        //cout << "hit enter to process next frame" << endl;
        //cin.get();
    }
#endif
}
catch (std::exception& e)
{
    cout << e.what() << endl;
}

