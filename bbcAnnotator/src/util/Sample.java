package util;

import java.nio.file.Path;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;

public class Sample {
	public String name;
	public boolean isTrain;
	public Path image;
	public Path text;
	public Path caption;
	// mutual information
	public int annotationMI = -1; // -1 is unknown
	// semantic correlation
	public int annotationSC = -3; // -3 is unknown
	// text, that was marked to have a particular high correlation with the image
	//public ArrayList<String> markedText = new ArrayList<String>();
	public ObservableList<String> markedText = FXCollections.observableArrayList();
	
	public String imageType = "Photograph";
	
}
