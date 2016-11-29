#include "mainwindow.h"
#include "ui_mainwindow.h"

using namespace cv;
using namespace std;

inline QImage  cvMatToQImage( const cv::Mat &inMat )
{
    switch ( inMat.type() )
    {
    // 8-bit, 4 channel
    case CV_8UC4:
    {
        QImage image( inMat.data, inMat.cols, inMat.rows, inMat.step, QImage::Format_RGB32 );

        return image;
    }

        // 8-bit, 3 channel
    case CV_8UC3:
    {
        QImage image( inMat.data, inMat.cols, inMat.rows, inMat.step, QImage::Format_RGB888 );

        return image.rgbSwapped();
    }

        // 8-bit, 1 channel
    case CV_8UC1:
    {
        static QVector<QRgb>  sColorTable;

        // only create our color table once

        QImage image( inMat.data, inMat.cols, inMat.rows, inMat.step, QImage::Format_Indexed8 );
        return image;
    }

    default:
    {
       break;
    }

    }

    return QImage();
}

inline QPixmap cvMatToQPixmap( const cv::Mat &inMat )
{
    return QPixmap::fromImage( cvMatToQImage( inMat ) );
}


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    init_form();

}

void MainWindow::init_form()
{
    main_directory = "/home/edwin/Desktop/Projects/CV 93 Project 1/";

    QDir myDir(main_directory);
    QStringList filesList = myDir.entryList(QDir::NoDotAndDotDot | QDir::System | QDir::Hidden  | QDir::AllDirs | QDir::Files, QDir::DirsFirst);

    for ( int i = 0 ; i < filesList.size() ; i++ )
    {
        ui->lst_names->addItem(filesList.at(i));
    }
}

double uniform()
{
    return (rand()/(float)0x7fff)-0.5;
}

struct RGB {
    uchar blue;
    uchar green;
    uchar red;  };


Mat MainWindow::filter_avg()
{
    Mat result;
    blur(image, result, Size(3,3));
    return result;
}

Mat MainWindow::filter_gaussian()
{
    Mat result;
    GaussianBlur( image, result, Size( 3, 3 ), 0, 0 );
    return result;
}

Mat MainWindow::filter_median()
{
    Mat result;
    int a = 3;
    QString g = ui->txt_sizem->toPlainText();
    a = g.toInt();

    medianBlur( image, result, a );
    return result;
}

Mat MainWindow::Add_Uniform_Noise()
{
    Mat result = image;
    for(int y = 0; y < image.rows; y++)
    {
        for(int x = 0; x < image.cols; x++)
        {
            uchar v = uniform();
            result.at<cv::Vec3b>(y,x)[0] = image.at<cv::Vec3b>(y,x)[0] + v * 0.4;
            result.at<cv::Vec3b>(y,x)[1] = image.at<cv::Vec3b>(y,x)[1] + v * 0.4;
            result.at<cv::Vec3b>(y,x)[2] = image.at<cv::Vec3b>(y,x)[2] + v * 0.4;
        }
    }

    return result;
}

Mat MainWindow::Add_salt_pepper_Noise( )
{
    float pa = 0.1;
    float pb = 0.05;

    Mat srcArr = image;
    RNG rng;
    int amount1=srcArr.rows*srcArr.cols*pa;
    int amount2=srcArr.rows*srcArr.cols*pb;
    for(int counter=0; counter<amount1; ++counter)
    {
        srcArr.at<uchar>(rng.uniform( 0,srcArr.rows), rng.uniform(0, srcArr.cols)) =0;
    }
    for (int counter=0; counter<amount2; ++counter)
    {
        srcArr.at<uchar>(rng.uniform(0,srcArr.rows), rng.uniform(0,srcArr.cols)) = 255;
    }

    return srcArr;
}

Mat MainWindow::Add_gaussian_Noise()
{
    double mean = 0;
    double sigma = 10;

    Mat srcArr = image;
    Mat NoiseArr = srcArr.clone();
    RNG rng;
    rng.fill(NoiseArr, RNG::NORMAL, mean,sigma);
    add(srcArr, NoiseArr, srcArr);
    return srcArr;
}

void MainWindow::preprocess_image()
{
    QString c = ui->txt_count->toPlainText();

    int a = c.toInt();

    for  (int i = 0 ; i < a ; i++ )
    {

        if ( ui->radio_grayscale->isChecked() )
            cv::cvtColor(image, image_output,cv::COLOR_BGR2GRAY);
        else
            if ( ui->radio_histogramequalization->isChecked() )
                cv::equalizeHist( image, image_output );
            else
                if ( ui->radio_calchitogram->isChecked())
                {
                    get_histogram();
                    image_output = image;
                }
                else if ( ui->radio_ContrastStretching->isChecked())
                {
                    cv::normalize(image, image_output, 200, 250, cv::NORM_MINMAX);
                }
                else if ( ui->radio_gaussiannoise->isChecked())
                {
                    image_output = Add_gaussian_Noise();
                }
                else if ( ui->radio_saltpapernoise->isChecked())
                {
                    image_output = Add_salt_pepper_Noise();
                }
                else if ( ui->radio_uniformnoise->isChecked())
                {
                    image_output = Add_Uniform_Noise();
                }
                else if ( ui->radio_filteravg->isChecked())
                {
                    image_output = filter_avg();
                }
                else if ( ui->radio_filtergaussian->isChecked())
                {
                    image_output = filter_gaussian();
                }
                else if ( ui->radio_filtermedian->isChecked())
                {
                    image_output = filter_median();
                }
                else
                    image_output = image;


        update_image_output();

        if ( i >= 1 )
        {
            save_step();
        }
    }

}

int computeOutput(int x, int r1, int s1, int r2, int s2)
{
    float result;
    if(0 <= x && x <= r1){
        result = s1/r1 * x;
    }else if(r1 < x && x <= r2){
        result = ((s2 - s1)/(r2 - r1)) * (x - r1) + s1;
    }else if(r2 < x && x <= 255){
        result = ((255 - s2)/(255 - r2)) * (x - r2) + s2;
    }
    return (int)result;
}

Mat MainWindow::stretching()
{
    Mat new_image = image.clone();

    for(int y = 0; y < image.rows; y++){
        for(int x = 0; x < image.cols; x++){
            for(int c = 0; c < 3; c++){
                int output = computeOutput(image.at<Vec3b>(y,x)[c], 70, 0, 140, 255);
                new_image.at<Vec3b>(y,x)[c] = saturate_cast<uchar>(output);
            }
        }
    }

    return new_image;
}

void MainWindow::get_histogram()
{
    int bins = 256;             // number of bins
    int nc = image.channels();    // number of channels
    vector<Mat> hist(nc);       // array for storing the histograms
    vector<Mat> canvas(nc);     // images for displaying the histogram
    int hmax[3] = {0,0,0};      // peak value for each histogram

    // The rest of the code will be placed here
    for (int i = 0; i < hist.size(); i++)
        hist[i] = Mat::zeros(1, bins, CV_32SC1);

    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            for (int k = 0; k < nc; k++)
            {
                uchar val = nc == 1 ? image.at<uchar>(i,j) : image.at<Vec3b>(i,j)[k];
                hist[k].at<int>(val) += 1;
            }
        }
    }

    for (int i = 0; i < nc; i++)
    {
        for (int j = 0; j < bins-1; j++)
            hmax[i] = hist[i].at<int>(j) > hmax[i] ? hist[i].at<int>(j) : hmax[i];
    }

    const char* wname[3] = { "blue", "green", "red" };
    Scalar colors[3] = { Scalar(255,0,0), Scalar(0,255,0), Scalar(0,0,255) };

    ui->lbl_hist_1->hide();
    ui->lbl_hist_2->hide();
    ui->lbl_hist_3->hide();

    for (int i = 0; i < nc; i++)
    {
        canvas[i] = Mat::ones(125, bins, CV_8UC3);

        for (int j = 0, rows = canvas[i].rows; j < bins-1; j++)
        {
            line(
                        canvas[i],
                        Point(j, rows),
                        Point(j, rows - (hist[i].at<int>(j) * rows/hmax[i])),
                        nc == 1 ? Scalar(200,200,200) : colors[i],
                        1, 8, 0
                        );
        }

        QPixmap imgIn = cvMatToQPixmap(canvas[i]);

        if ( i == 0 )
        {
            ui->lbl_hist_1->setPixmap(imgIn);
            ui->lbl_hist_1->setScaledContents(true);
            ui->lbl_hist_1->show();
        }

        if ( i == 1 )
        {
            ui->lbl_hist_2->setPixmap(imgIn);
            ui->lbl_hist_2->setScaledContents(true);
            ui->lbl_hist_2->show();
        }

        if ( i == 2 )
        {
            ui->lbl_hist_3->setPixmap(imgIn);
            ui->lbl_hist_3->setScaledContents(true);
            ui->lbl_hist_3->show();
        }

        //imshow(nc == 1 ? "value" : wname[i], canvas[i]);
    }
}

void MainWindow::find_contours()
{
    //Mat output;

    std::vector<std::vector<cv::Point> > contours;
    cv::Mat contourOutput = image.clone();
    cv::findContours( contourOutput, contours, RETR_LIST, CHAIN_APPROX_TC89_L1 );

    cv::Mat contourImage(image.size(), CV_8UC3, cv::Scalar(0,0,0));
    cv::Scalar colors[10];
    colors[0] = cv::Scalar(255, 0, 0);
    colors[1] = cv::Scalar(255, 255, 0);
    colors[2] = cv::Scalar(0, 255, 0);
    colors[3] = cv::Scalar(0, 255, 255);
    colors[4] = cv::Scalar(0, 0, 255);
    colors[5] = cv::Scalar(255, 0, 255);
    colors[6] = cv::Scalar(255, 255, 255);
    colors[7] = cv::Scalar(100, 0, 0);
    colors[8] = cv::Scalar(0, 100, 0);
    colors[9] = cv::Scalar(0, 0, 100);

    for (size_t idx = 0; idx < contours.size(); idx++) {
        cv::drawContours(contourImage, contours, idx, colors[idx % 10],-1);
    }

    image_output = contourImage;
    update_image_output();
}

void MainWindow::save_step()
{
    image = image_output.clone();
}

void MainWindow::cany_edge()
{
    Canny( image, image_output, ui->slider_cany1->value(), ui->slider_cany2->value(), 3 );
    update_image_output();
}

void MainWindow::update_image()
{
    QPixmap imgIn = cvMatToQPixmap(image);

    ui->label->setPixmap(imgIn);
    ui->label->setScaledContents(true);
    ui->label->show();
}

void MainWindow::update_image_output()
{
    QPixmap imgIn = cvMatToQPixmap(image_output);

    ui->label->setPixmap(imgIn);
    ui->label->setScaledContents(true);
    ui->label->show();
}

void MainWindow::binary()
{
    cv::threshold(image, image_output, ui->slider_binary->value(), 255, THRESH_BINARY);
    update_image_output();
}

void MainWindow::load_image()
{

    QString selected_item = "none";


    QListWidgetItem* item = ui->lst_names->currentItem();
    if ( NULL == item ) return;

    selected_item = ui->lst_names->currentItem()->text();

    QString path = main_directory + selected_item;

    original_image = cv::imread(path.toStdString());
    image = original_image;
    update_image();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_btn_loadimage_clicked()
{
    load_image();
}

void MainWindow::on_btn_processimage_clicked()
{
    preprocess_image();
}

void MainWindow::on_btn_contours_clicked()
{
    find_contours();
}

void MainWindow::on_btn_cany_clicked()
{
    cany_edge();
}

void MainWindow::on_btn_binary_clicked()
{
    binary();
}

void MainWindow::on_btn_setp_clicked()
{
    save_step();
}

void MainWindow::on_btn_show_clicked()
{
    update_image();
}

struct pixel_val
{
public:
    int X;
    int Y;
};

std::vector<pixel_val> region_list;

int dist_calc(cv::Vec3b c1,cv::Vec3b c2,int threshold)
{
    int val = 0;
    float a = c1[0] - c2[0];
    float b = c1[1] - c2[1];
    float c = c1[2] - c2[2];

    double x = sqrt(a*a + b*b + c*c);

    if ( x > threshold ) return 0;
    if ( x < threshold ) return 1;

    return val;
}

bool check_inbox(pixel_val item , int w , int h)
{
    if ( item.X < 0) return false;
    if ( item.Y < 0) return false;
    if ( item.X > w) return false;
    if ( item.Y > h) return false;
    return true;
}

Mat *wimage;
int rows;
int cols;

bool check_isnotinregion(pixel_val item)
{
    for ( int i = 0 ; i < region_list.size() ; i++ )
    {
        pixel_val l = region_list.at(i);
        if ( item.X == l.X && item.Y == l.Y )
            return false;
    }

    cv::Vec3b intensity1 = wimage->at<cv::Vec3b>(item.X,item.Y);
    if ( intensity1[0] != 0 ) return false;
    if ( intensity1[1] != 0 ) return false;
    if ( intensity1[2] != 0 ) return false;

    return true;
}

void MainWindow::region_growing_process()
{

    //create a grid and use growing for eache seed

    cv::Size s = image.size();
    rows = s.height;
    cols = s.width;

    wimage = new cv::Mat(rows, cols , image.type(), cv::Scalar(0,0,0));

    for ( int i = 0 ; i < rows ; i += 10)
    {
        for ( int j = 0 ; j < cols ; j += 10)
        {
            region_list.clear();
            region_growing(i,j,3);
        }
    }


}

void MainWindow::region_growing(int seedx,int seedy,int threshold)
{
    cv::Vec3b intensity1 = image.at<cv::Vec3b>(seedx,seedy);

    cv::Vec3b intensity00;
    intensity00[0] = 250;
    intensity00[1] = 250;
    intensity00[2] = 250;

    pixel_val seed;
    seed.X = seedx;
    seed.Y = seedy;
    region_list.push_back(seed);

    int region_item = 0;

    pixel_val r1 , r2 , r3 , r4 , r5 , r6 , r7 , r8;

    for ( ; region_item < region_list.size() && region_list.size() < 5000 ; region_item++ )
    {
        pixel_val item = region_list.at(region_item);

        cv::Vec3b intensity1 = image.at<cv::Vec3b>(item.X,item.Y);

        r1.X = item.X - 1;
        r1.Y = item.Y - 1;

        r2.X = item.X ;
        r2.Y = item.Y - 1;

        r3.X = item.X + 1;
        r3.Y = item.Y - 1;

        r4.X = item.X - 1;
        r4.Y = item.Y;

        r5.X = item.X + 1;
        r5.Y = item.Y ;

        r6.X = item.X - 1;
        r6.Y = item.Y + 1;

        r7.X = item.X;
        r7.Y = item.Y + 1;

        r8.X = item.X + 1;
        r8.Y = item.Y + 1;

        if ( check_inbox(r1,rows,cols) && check_isnotinregion(r1))
        {
            cv::Vec3b inte = image.at<cv::Vec3b>(r1.X, r1.Y);
            int r = dist_calc(inte,intensity1,threshold);
            if ( r == 1 )
            {
                region_list.push_back(r1);
            }
        }

        if ( check_inbox(r2,rows,cols) && check_isnotinregion(r2))
        {
            cv::Vec3b inte = image.at<cv::Vec3b>(r2.X, r2.Y);
            int r = dist_calc(inte,intensity1,threshold);
            if ( r == 1 )
            {
                region_list.push_back(r2);
            }
        }

        if ( check_inbox(r3,rows,cols) && check_isnotinregion(r3))
        {
            cv::Vec3b inte = image.at<cv::Vec3b>(r3.X, r3.Y);
            int r = dist_calc(inte,intensity1,threshold);
            if ( r == 1 )
            {
                region_list.push_back(r3);
            }
        }

        if ( check_inbox(r4,rows,cols) && check_isnotinregion(r4))
        {
            cv::Vec3b inte = image.at<cv::Vec3b>(r4.X, r4.Y);
            int r = dist_calc(inte,intensity1,threshold);
            if ( r == 1 )
            {
                region_list.push_back(r4);
            }
        }

        if ( check_inbox(r5,rows,cols) && check_isnotinregion(r5))
        {
            cv::Vec3b inte = image.at<cv::Vec3b>(r5.X, r5.Y);
            int r = dist_calc(inte,intensity1,threshold);
            if ( r == 1 )
            {
                region_list.push_back(r5);
            }
        }

        if ( check_inbox(r6,rows,cols) && check_isnotinregion(r6))
        {
            cv::Vec3b inte = image.at<cv::Vec3b>(r6.X, r6.Y);
            int r = dist_calc(inte,intensity1,threshold);
            if ( r == 1 )
            {
                region_list.push_back(r6);
            }
        }

        if ( check_inbox(r7,rows,cols) && check_isnotinregion(r7))
        {
            cv::Vec3b inte = image.at<cv::Vec3b>(r7.X, r7.Y);
            int r = dist_calc(inte,intensity1,threshold);
            if ( r == 1 )
            {
                region_list.push_back(r7);
            }
        }

        if ( check_inbox(r8,rows,cols) && check_isnotinregion(r8))
        {
            cv::Vec3b inte = image.at<cv::Vec3b>(r8.X, r8.Y);
            int r = dist_calc(inte,intensity1,threshold);
            if ( r == 1 )
            {
                region_list.push_back(r8);
            }
        }

    }

    for ( int i = 0 ; i < region_list.size() ; i++ )
    {
        pixel_val l = region_list.at(i);
        if ( intensity1[0] == 0 && intensity1[1] == 0 && intensity1[2] == 0 )
            wimage->at<cv::Vec3b>(l.X, l.Y) = intensity00;
        else
            wimage->at<cv::Vec3b>(l.X, l.Y) = intensity1;
    }

    image_output = wimage->clone();
    update_image_output();
}

int kernel_size=21;
int pos_sigma= 5;
int pos_lm = 50;
int pos_th = 0;
int pos_psi = 90;
cv::Mat src_f;
cv::Mat dest;

cv::Mat mkKernel(int ks, double sig, double th, double lm, double ps)
{
    int hks = (ks-1)/2;
    double theta = th*CV_PI/180;
    double psi = ps*CV_PI/180;
    double del = 2.0/(ks-1);
    double lmbd = lm;
    double sigma = sig/ks;
    double x_theta;
    double y_theta;
    cv::Mat kernel(ks,ks, CV_32F);
    for (int y=-hks; y<=hks; y++)
    {
        for (int x=-hks; x<=hks; x++)
        {
            x_theta = x*del*cos(theta)+y*del*sin(theta);
            y_theta = -x*del*sin(theta)+y*del*cos(theta);
            kernel.at<float>(hks+y,hks+x) = (float)exp(-0.5*(pow(x_theta,2)+pow(y_theta,2))/pow(sigma,2))* cos(2*CV_PI*x_theta/lmbd + psi);
        }
    }
    return kernel;
}



void Process(int , void *)
{
    double sig = pos_sigma;
    double lm = 0.5+pos_lm/100.0;
    double th = pos_th;
    double ps = pos_psi;

    cv::Mat kernel = mkKernel(kernel_size, sig, th, lm, ps);

    cv::filter2D(src_f, dest, CV_32F, kernel);
    cv::imshow("Process window", dest);

    cv::Mat Lkernel(kernel_size*20, kernel_size*20, CV_32F);
    cv::resize(kernel, Lkernel, Lkernel.size());
    Lkernel /= 2.;
    Lkernel += 0.5;
    cv::imshow("Kernel", Lkernel);
    cv::Mat mag;
    cv::pow(dest, 2.0, mag);
    cv::imshow("Mag", mag);

}

void MainWindow::gabor_filter()
{

    //cv::imshow("Src", image);
    cv::Mat src;
    cv::cvtColor(image, src, cv::COLOR_BGR2GRAY);
    src.convertTo(src_f, CV_32F, 1.0/255, 0);
    if (!kernel_size%2)
    {
        kernel_size+=1;
    }
    cv::namedWindow("Process window", 1);
    cv::createTrackbar("Sigma", "Process window", &pos_sigma, kernel_size, Process);
    cv::createTrackbar("Lambda", "Process window", &pos_lm, 100, Process);
    cv::createTrackbar("Theta", "Process window", &pos_th, 180, Process);
    cv::createTrackbar("Psi", "Process window", &pos_psi, 360, Process);
    Process(0,0);
    cv::waitKey(0);

}

void MainWindow::k_means()
{
    //step 1
    cv::Mat samples(image.total(), 3, CV_32F);
    float *samples_ptr = samples.ptr<float>(0);

    for( int row = 0; row != image.rows; ++row)
    {
        uchar *src_begin = image.ptr<uchar>(row);
        uchar *src_end = src_begin + image.cols * image.channels();
        //auto samples_ptr = samples.ptr<float>(row * src.cols);
        while(src_begin != src_end){
            samples_ptr[0] = src_begin[0];
            samples_ptr[1] = src_begin[1];
            samples_ptr[2] = src_begin[2];
            samples_ptr += 3; src_begin +=3;
        }
    }

    //step 2
    int clusterCount = 5;
    cv::Mat labels;
    int attempts = 5;
    cv::Mat centers;
    cv::kmeans(samples, clusterCount, labels,cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,10, 0.01), attempts, cv::KMEANS_PP_CENTERS, centers);

    //step 3 : map the centers to the output
    cv::Mat new_image(image.size(), image.type());
    for( int row = 0; row != image.rows; ++row){
        uchar * new_image_begin = new_image.ptr<uchar>(row);
        uchar * new_image_end = new_image_begin + new_image.cols * 3;
        int * labels_ptr = labels.ptr<int>(row * image.cols);

        while(new_image_begin != new_image_end){
            int const cluster_idx = *labels_ptr;
            float * centers_ptr = centers.ptr<float>(cluster_idx);
            new_image_begin[0] = centers_ptr[0];
            new_image_begin[1] = centers_ptr[1];
            new_image_begin[2] = centers_ptr[2];
            new_image_begin += 3; ++labels_ptr;
        }
    }

    cv::Mat binary;
    cv::Canny(new_image, binary, 30, 90);

    image_output = new_image;
    update_image_output();


}

void MainWindow::on_btn_contours_2_clicked()
{
    k_means();
}

void MainWindow::on_btn_contours_3_clicked()
{
    gabor_filter();
}

void MainWindow::on_btn_contours_4_clicked()
{
    double sig = pos_sigma;
    double lm = 0.5+pos_lm/100.0;
    double th = pos_th;
    double ps = pos_psi;
    cv::Mat kernel = mkKernel(kernel_size, sig, th, lm, ps);
    cv::filter2D(src_f, dest, CV_32F, kernel);
    //cv::imshow("Process window", dest);
    cv::Mat Lkernel(kernel_size*20, kernel_size*20, CV_32F);
    cv::resize(kernel, Lkernel, Lkernel.size());
    Lkernel /= 2.;
    Lkernel += 0.5;
    //cv::imshow("Kernel", Lkernel);
    cv::Mat mag;
    cv::pow(dest, 2.0, mag);
    //cv::imshow("Mag", mag);
    int type = mag.type();
    string r;

     uchar depth = type & CV_MAT_DEPTH_MASK;
     uchar chans = 1 + (type >> CV_CN_SHIFT);

     switch ( depth ) {
       case CV_8U:  r = "8U"; break;
       case CV_8S:  r = "8S"; break;
       case CV_16U: r = "16U"; break;
       case CV_16S: r = "16S"; break;
       case CV_32S: r = "32S"; break;
       case CV_32F: r = "32F"; break;
       case CV_64F: r = "64F"; break;
       default:     r = "User"; break;
     }

     r += "C";
     r += (chans+'0');

     Mat Temp;
     mag.convertTo(Temp, CV_8UC3);


    image = Temp;
    update_image();
}
