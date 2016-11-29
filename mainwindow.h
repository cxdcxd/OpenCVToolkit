#ifndef MAINWINDOW_H
#define MAINWINDOW_H

//--------------------------------------------- QT
#include <QMainWindow>
#include <QDir>
#include <QString>
#include <math.h>

//--------------------------------------------- OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);

    cv::Mat image_output;
    void update_image_output();

    ~MainWindow();

private slots:
    void on_btn_loadimage_clicked();

    void on_btn_processimage_clicked();

    void on_btn_contours_clicked();

    void on_btn_cany_clicked();

    void on_btn_binary_clicked();

    void on_btn_setp_clicked();

    void on_btn_show_clicked();

    void on_btn_contours_2_clicked();



    void on_btn_contours_3_clicked();

    void on_btn_contours_4_clicked();

private:
    Ui::MainWindow *ui;
   
    void init_form();
    void preprocess_image();
     void load_image();
    void update_image();

    void get_histogram();
    cv::Mat stretching();
    cv::Mat Add_gaussian_Noise();
    cv::Mat Add_salt_pepper_Noise();
    cv::Mat Add_Uniform_Noise();

    cv::Mat filter_avg();
    cv::Mat filter_median();
    cv::Mat filter_gaussian();

    void find_contours();
    void cany_edge();
    void binary();
    void save_step();
    void k_means();
    void region_growing(int sx,int sy,int th);
    void region_growing_process();
    void gabor_filter();

    cv::Mat original_image; //loaded image
    cv::Mat image; //current image

    QString main_directory;
};

#endif // MAINWINDOW_H
