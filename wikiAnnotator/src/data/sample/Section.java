package data.sample;

import java.util.ArrayList;

public class Section {
	
	public Section() {
		subsections = new ArrayList<>();
		images = new ArrayList<>();
		lists = new ArrayList<>();
		keyphrases = new ArrayList<>();
	}
	
	public String title;
	public String text;
	public ArrayList<Section> subsections;
	public ArrayList<Image> images;
	public ArrayList<ArrayList<String>> lists;
	public ArrayList<String> keyphrases;
	
	public String toString() {
		return this.title;
	}
}
