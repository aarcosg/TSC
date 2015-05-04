package us.idinfor.tsclassification;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.CvSVM;
import org.opencv.objdetect.HOGDescriptor;

public class HogClassificationTest {

	public static void main(String[] args) {
		
		System.loadLibrary( Core.NATIVE_LIBRARY_NAME );
		Mat img = Highgui.imread("stop.jpg", Highgui.CV_LOAD_IMAGE_GRAYSCALE);
		Imgproc.resize(img, img, new Size(40,40));
		   
		MatOfFloat descriptors = new MatOfFloat();
		   
		HOGDescriptor hog = new HOGDescriptor(new Size(40,40), new Size(8,8), new Size(4,4), new Size(4,4), 9);
		hog.compute(img, descriptors);
		   
		System.out.println("Found descriptors: " + descriptors.size() + "\n" + hog.getDescriptorSize());	

		CvSVM SVM = new CvSVM();
		SVM.load("svmhog3.xml");
		float classId = SVM.predict(descriptors);
		System.out.println("ClassID = " + classId);

	}

}
