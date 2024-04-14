#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main() {
    CascadeClassifier faceCascade, eyesCascade, smileCascade;
    faceCascade.load("haarcascade_frontalface_alt.xml");
    eyesCascade.load("haarcascade_eye_tree_eyeglasses.xml");
    smileCascade.load("haarcascade_smile.xml");


    VideoCapture cap("C:/Users/User/Desktop/ZUA.mp4");

    if (!cap.isOpened())
    {
        cout << "Error" << endl;
        return -1;
    }

    int width = cap.get(CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);

    int video = VideoWriter::fourcc('X', 'V', 'I', 'D');
    VideoWriter videoOutput("C:/Users/User/Desktop/output.mp4", video, 20, Size(width, height));


    if (!videoOutput.isOpened())
    {
        cout << "Error with file" << endl;
        return -1;
    }


    while (true)
    {
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            cout << "End" << endl;
            break;
        }

        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);

        vector<Rect> faces, eyes, smiles;
        faceCascade.detectMultiScale(gray, faces, 2, 3, 0, Size(20, 20));

        for (const auto& face : faces) {
            rectangle(frame, face, Scalar(58, 64, 224), 2);

            eyesCascade.detectMultiScale(gray(face), eyes, 3, 2, 0, Size(5, 5));
            for (const auto& eye : eyes) {
                Point center(eye.x + eye.width / 2, eye.y + eye.height / 2);
                int radius = cvRound((eye.width + eye.height) * 0.25);
                circle(frame(face), center, radius, Scalar(63, 181, 110), 2);
            }

            smileCascade.detectMultiScale(gray(face), smiles, 1.565, 30, 0, Size(30, 30));
            for (const auto& smile : smiles) {
                rectangle(frame(face), smile, Scalar(181, 63, 169), 2);
            }

            blur(frame(face), frame(face), Size(3, 3));
        }
        imshow("Video", frame);
        videoOutput.write(frame);

        if (waitKey(25) == 'q')
            break;
    }

    cap.release();
    videoOutput.release();
    destroyAllWindows();

    return 0;
}
