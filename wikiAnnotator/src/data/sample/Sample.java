package data.sample;

import java.util.ArrayList;

public class Sample {
	public Sample() {
		sections = new ArrayList<>();
		references = new ArrayList<>();
		keyphrases = new ArrayList<>();
		categories = new ArrayList<>();
		images = new ArrayList<>();		
	}
	
	public int id;
	public String title;
	public ArrayList<Section> sections;
	public ArrayList<String> references;
	public String summary;
	public ArrayList<String> keyphrases;
	public String url;
	public ArrayList<String> categories;
	public ArrayList<Image> images;	
}
