///
///  Assignment 2
///  Pedestrian Detection
///
#include <opencv2/opencv.hpp>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <fstream>

#include "hog.h"
#include "ROC.h"
using namespace std;



int main(int argc, char* argv[]) {
	
	//// parse command line options
	boost::program_options::variables_map pom;
	{
		namespace po = boost::program_options;
		po::options_description pod(string("Allowed options for ")+argv[0]);
		pod.add_options() 
			("help,h", "produce this help message")
			("gui,g", "Enable the GUI");

		po::store(po::command_line_parser( argc, argv ).options(pod).run(), pom);
		po::notify(pom);

		if (pom.count("help")) {
			cout << "Usage:" << endl <<  pod << "\n";
			return 0;
		}
	}

	//// TRAINING
    /// create person classification model instance
    HOG model;
	
	namespace fs = boost::filesystem; 
	/// get training image filenames
	vector<pair<string,bool>> trainImgs;
	{
		for (fs::directory_iterator it(fs::path("data/train/p")); it!=fs::directory_iterator(); it++)
			if (is_regular_file(*it))
				trainImgs.push_back({it->path().filename().string(),1});
		for (fs::directory_iterator it(fs::path("data/train/n")); it!=fs::directory_iterator(); it++)
			if (is_regular_file(*it))
				trainImgs.push_back({it->path().filename().string(),0});
    } 
    /// train model with all images in the train folder
	cout << "Start Training" << endl;
	model.startTraining();
	
	random_shuffle(trainImgs.begin(), trainImgs.end());
	for (auto &&f : trainImgs) {
		cout << "Training on Image " << std::string()+"data/train/"+"np"[f.second]+"/"+f.first << endl;
		cv::Mat3b img = cv::imread(std::string()+"data/train/"+"np"[f.second]+"/"+f.first,-1);
		model.train( img, f.second );
	}
	
	cout << "Finish Training" << endl;
	model.finishTraining();
	
	
	//// VALIDATION

	/// get validation image filenames
	vector<pair<string,bool>> validationImgs;
	{
		for (fs::directory_iterator it(fs::path("data/validation/p")); it!=fs::directory_iterator(); it++)
			if (is_regular_file(*it))
				validationImgs.push_back({it->path().filename().string(),1});
		for (fs::directory_iterator it(fs::path("data/validation/n")); it!=fs::directory_iterator(); it++)
			if (is_regular_file(*it))
				validationImgs.push_back({it->path().filename().string(),0});
    } 
    
	
    /// test model with all images in the test folder, 
	ROC<double> roc;
	random_shuffle(validationImgs.begin(), validationImgs.end());
	for (auto &&f : validationImgs) {
		cv::Mat3b img = cv::imread(std::string()+"data/validation/"+"np"[f.second]+"/"+f.first,-1);
		double hyp = model.classify(img);
		roc.add(f.second, hyp);
		cout << "Validating Image " << f.second << " " << hyp << " " << std::string()+"data/validation/"+"np"[f.second]+"/"+f.first << endl;
	}
	
	/// After testing, update statistics and show results
	roc.update();
	
	cout << "Overall F1 score: " << roc.F1 << endl;
	
	/// Display final result if desired
	if (pom.count("gui")) {
		cv::imshow("ROC", roc.draw());
		cv::waitKey(0);
	}

	//// TESTING

	/// CLASSIFY TEST DATA:
	ifstream iss("data/testFiles.txt");
	ofstream oss("result.txt");
	string filename;
	while (iss >> filename) {
		cv::Mat3b img = cv::imread("data/test/" + filename,-1);
		if (img.rows==0) { cerr << " ERROR: " << filename << endl; continue; }
		double hyp = model.classify(img);
		cout << "Test Image " << filename << ": " << hyp << endl;
		oss << hyp << std::endl;
	}
}

