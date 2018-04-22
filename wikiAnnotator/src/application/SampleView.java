package application;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;

import org.json.JSONException;
import org.json.JSONObject;

import annotation.Annotation;
import data.Backup;
import data.DatasetHandler;
import data.sample.Image;
import data.sample.Sample;
import data.sample.Section;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.geometry.Pos;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.ListView;
import javafx.scene.control.Pagination;
import javafx.scene.control.RadioButton;
import javafx.scene.control.Tab;
import javafx.scene.control.TabPane;
import javafx.scene.control.TabPane.TabClosingPolicy;
import javafx.scene.control.TextArea;
import javafx.scene.control.Toggle;
import javafx.scene.control.ToggleGroup;
import javafx.scene.control.Tooltip;
import javafx.scene.control.TreeItem;
import javafx.scene.control.TreeView;
import javafx.scene.image.ImageView;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyEvent;
import javafx.scene.input.ScrollEvent;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.scene.web.WebEngine;
import javafx.scene.web.WebView;
import javafx.stage.Stage;
import gui.ImageViewPane;

public class SampleView extends BorderPane {

    @FXML
    private VBox imageCaptionView;

    @FXML
    private WebView wikiPageView;
    
    @FXML
    private TreeView<Section> sectionList;
    private javafx.scene.image.Image imgIcon = new javafx.scene.image.Image(getClass().getClassLoader().getResourceAsStream("icons/icon-image.png"));
    
    @FXML
    private Label articleTitle;
    
    @FXML
    private WebView textView;
    private WebEngine textViewEngine;
    
    @FXML
    private Pagination imgPagination;
    
    @FXML
    private BorderPane annotationView;

    @FXML
    private Label annoTitle;
    
    @FXML
    private HBox radioBtnsHBoxMI;
    private ToggleGroup radioGroupMI;
    private RadioButton[] radioBtnsMI = {null,null,null,null,null,null,null,null,null};

    @FXML
    private HBox radioBtnsHBoxSC;
    private ToggleGroup radioGroupSC;
    private RadioButton[] radioBtnsSC = {null,null,null,null,null,null};

    @FXML
    private ComboBox<String> markedTextCB;

    @FXML
    private Button addBtn;

    @FXML
    private Button removeBtn;
    
    @FXML
    private CheckBox checkboxValid;

    @FXML
    private ComboBox<String> imageTypeCB;
    
    @FXML
    private ListView<String> listKeyphrases;
    
    @FXML
    private VBox listsVBox;
    
    // sample belonging to this view
    private Sample sample;
    
    private Annotation currAnno = null;
    
    // if control key is pressed
    private boolean controlPressed = false;
    
    private Stage stage;
    
    private static final String[] imgTypes = { // TODO
    		"Chart - Bar Chart",
    		"Chart - Flow Chart",
    		"Chart - Histogram",
    		"Chart - Line Chart",
    		"Chart - Pie Chart",
    		"Chart - Tree Chart",
    		"Chart - Other",
    		"Artistic Drawing",
    		"Technical Drawing",
    		"Graph",
    		"Map",
    		"Photograph",
    		"Table"
    };
    
    // script to get selected text from webview
 	// source: https://gist.github.com/jewelsea/7819195
 	public static final String SELECT_TEXT =
             "(function getSelectionText() {\n" +
             "    var text = \"\";\n" +
             "    if (window.getSelection) {\n" +
             "        text = window.getSelection().toString();\n" +
             "    } else if (document.selection && document.selection.type != \"Control\") {\n" +
             "        text = document.selection.createRange().text;\n" +
             "    }\n" +
             "    if (window.getSelection) {\n" +
             "      if (window.getSelection().empty) {  // Chrome\n" +
             "        window.getSelection().empty();\n" +
             "      } else if (window.getSelection().removeAllRanges) {  // Firefox\n" +
             "        window.getSelection().removeAllRanges();\n" +
             "      }\n" +
             "    } else if (document.selection) {  // IE?\n" +
             "      document.selection.empty();\n" +
             "    }" +
             "    return text;\n" +
             "})()";
    
    public SampleView(Stage stage) {
    	try {
            FXMLLoader loader = new FXMLLoader(getClass().getResource("SampleView.fxml"));
            loader.setController(this);
            loader.setRoot(this);
            loader.load();
        } catch (IOException exc) {
        	exc.printStackTrace();
        }	
    	this.stage = stage;
    }
    
    public void setSample(int index) {
    	this.sample = DatasetHandler.getInstance().getSample(index);
    	articleTitle.setText(this.sample.title);
    	
    	// display webpage
    	WebEngine webEngine = wikiPageView.getEngine();
		webEngine.load(this.sample.url);
		
		// fill section tree view
		
		// we have to treat the whole article as a special section
		Section article = this.generateArticleSection();
		
		TreeItem<Section> articleItem;
		if (secHasImages(article)) {
			articleItem = new TreeItem<>(article, getImgIcon());
		} else 
			articleItem = new TreeItem<>(article);
		articleItem.setExpanded(true);
		sectionList.setRoot(articleItem);
		
		this.addSectionList(this.sample.sections, articleItem);
		
		// change section event
		sectionList.getSelectionModel().selectedItemProperty()
        .addListener(new ChangeListener<TreeItem<Section>>() {
            @Override
            public void changed(
                    ObservableValue<? extends TreeItem<Section>> observable,
                    TreeItem<Section> old_val, TreeItem<Section> new_val) {
            	displaySection(new_val.getValue());
            }
        });
		
		this.textViewEngine = this.textView.getEngine();
		
		// mutual information
        this.radioGroupMI = new ToggleGroup();  

        radioBtnsMI[0] = new RadioButton("Unknown");
        radioBtnsMI[0].setToggleGroup(this.radioGroupMI);
        radioBtnsMI[0].setUserData(-1);
        radioBtnsMI[0].setTooltip(new Tooltip("If annotation is not known yet."));
        
        for (int i = 0; i < 5; i++) {
            radioBtnsMI[i+1] = new RadioButton(""+(i/4.0));
            radioBtnsMI[i+1].setToggleGroup(this.radioGroupMI);
            radioBtnsMI[i+1].setUserData(i);
            radioBtnsMI[i+1].setTooltip(new Tooltip("TODO"));
            radioBtnsHBoxMI.getChildren().add(radioBtnsMI[i+1]);
        } 
        
        radioBtnsHBoxMI.getChildren().add(radioBtnsMI[0]);
        
        // radiobutton listener
        this.radioGroupMI.selectedToggleProperty().addListener(new ChangeListener<Toggle>() {
			public void changed(ObservableValue<? extends Toggle> ov, Toggle old_toggle, Toggle new_toggle) {
				if (radioGroupMI.getSelectedToggle() != null) {
					currAnno.annotationMI = (int) radioGroupMI.getSelectedToggle().getUserData();
					Backup.getInstance().backupAnnotation(currAnno);
				}
			}
		});
        
        // semantic correlation
        this.radioGroupSC = new ToggleGroup();

        radioBtnsSC[0] = new RadioButton("Unknown");
        radioBtnsSC[0].setToggleGroup(this.radioGroupSC);
        radioBtnsSC[0].setUserData(-3);
        radioBtnsSC[0].setTooltip(new Tooltip("If annotation is not known yet."));
        
        for (int i = -2; i < 3; i++) {
        	radioBtnsSC[i+3] = new RadioButton(""+(i/2.0));
        	radioBtnsSC[i+3].setToggleGroup(this.radioGroupSC);
            radioBtnsSC[i+3].setUserData(i);
            radioBtnsSC[i+3].setTooltip(new Tooltip("TODO"));
            radioBtnsHBoxSC.getChildren().add(radioBtnsSC[i+3]);
        } 
        
        radioBtnsHBoxSC.getChildren().add(radioBtnsSC[0]);
                
        // radiobutton listener
        this.radioGroupSC.selectedToggleProperty().addListener(new ChangeListener<Toggle>() {
			public void changed(ObservableValue<? extends Toggle> ov, Toggle old_toggle, Toggle new_toggle) {
				if (radioGroupSC.getSelectedToggle() != null) {
					currAnno.annotationSC = (int) radioGroupSC.getSelectedToggle().getUserData();
					Backup.getInstance().backupAnnotation(currAnno);
				}
			}
		});
        
        this.checkboxValid.setTooltip(new Tooltip("The dataset is generated by ignoring all additional information such as tables, lists, formula, etc.\nTherefore, it might be that the supplied text is not sufficient to have a relation to the text"));
		
        ObservableList<String> imageTypes = FXCollections.observableArrayList(imgTypes);
        this.imageTypeCB.setItems(imageTypes);
        this.imageTypeCB.setOnAction((event) -> {
            String selectedType = this.imageTypeCB.getSelectionModel().getSelectedItem();
            currAnno.imageType = selectedType;
            Backup.getInstance().backupAnnotation(currAnno);
        });
        
    	this.displaySection(article);
    	
    	stage.getScene().setOnKeyPressed(new EventHandler<KeyEvent>() {
            @Override
            public void handle(KeyEvent event) {
            	if (KeyCode.CONTROL == event.getCode())
    				controlPressed = true;
            }
        });
    	
    	stage.getScene().setOnKeyReleased(new EventHandler<KeyEvent>() {
            @Override
            public void handle(KeyEvent event) {
            	if (KeyCode.CONTROL == event.getCode())
    				controlPressed = false;
            }
        });
    }
    
    private void displaySection(Section sec) {
    	this.textViewEngine.loadContent(sec.text);
    	this.initImgPagination(sec);
    	
    	int ROW_HEIGHT = 24;
    	ObservableList<String> items = FXCollections.observableArrayList ();
		for (String kp : sec.keyphrases)
			items.add(kp);
		this.listKeyphrases.setItems(items);		
		this.listKeyphrases.setPrefHeight(items.size() * ROW_HEIGHT + 2);
		
		for (ArrayList<String> li : sec.lists) {
			ListView<String> listListItems = new ListView<>();
			items = FXCollections.observableArrayList ();
			for (String item : li)
				items.add(item);
			listListItems.setItems(items);		
			listListItems.setPrefHeight((items.size()+1) * ROW_HEIGHT + 2);
			listsVBox.getChildren().add(listListItems);
		}
    }
    
    private void addSectionList(ArrayList<Section> list, TreeItem<Section> root) {
    	for (Section sec : list) {
    		TreeItem<Section> ti;
    		if (secHasImages(sec))
    			ti = new TreeItem<>(sec, getImgIcon());
    		else 
    			ti = new TreeItem<>(sec);
    		ti.setExpanded(true);
    		root.getChildren().add(ti);
    		
    		if (sec.subsections.size() > 0) 
    			this.addSectionList(sec.subsections, ti);
    	}
    }
    
    // if section has images in its list with actual image files
    private boolean secHasImages(Section sec) {
    	for (Image img : sec.images) {
    		if (img.imgfile != null)
    			return true;
    	}
    	return false;
    }
    
    private ImageView getImgIcon() {
    	ImageView icon =  new ImageView(imgIcon);
    	icon.setPreserveRatio(true);
    	icon.setFitHeight(16);
    	return icon;
    }
    
    private void sectionsToHTML(ArrayList<Section> list, StringBuilder content, int depth) {
    	String prefix = "";
    	int d = depth;
    	while (d-- > 0)
    		prefix += "-";
    	
    	for (Section sec : list) {
    		content.append("<h2>" + prefix + sec.title + "</h2>\n");
    		content.append("<p>" + sec.text + "</p>\n");
    		sectionsToHTML(sec.subsections, content, depth+1);
    	}
    }
    
    // generate a dummy section object for the whole article
    private Section generateArticleSection() {
    	Section article = new Section();
		article.title = "Article";
		
		StringBuilder content = new StringBuilder();
		content.append("<p>" + this.sample.summary + "</p>\n");
		this.sectionsToHTML(this.sample.sections, content, 0);
		article.text = content.toString();
		
		article.images = this.sample.images;
		article.keyphrases = this.sample.keyphrases;
		
		return article;
    }
    
    private void initImgPagination(Section sec) {
    	annotationView.setVisible(false);
    	
    	ArrayList<Image> validImg = new ArrayList<>();
    	for (Image img : sec.images) {
    		if (img.imgfile != null)
    			validImg.add(img);
    	}
    	if (validImg.size() == 0) {
    		imgPagination.setVisible(false);
    	} else {
    		imgPagination.setVisible(true);
	    	imgPagination.setPageCount(validImg.size());
			this.imgPagination.setPageFactory((Integer pageIndex) -> {
				currAnno = null;
	            if (pageIndex >= validImg.size()) {
	            	return null;
	            } else {
	            	Image img = validImg.get(pageIndex);
	            	if (img.imgfile != null) {
	            		currAnno = Backup.getInstance().getAnnotation(sample.id, sec.title, img.name);
	            		showAnnotation();
	            		return createImgView(img);
	            	} else 
	            		return null;
	            }
	        });
    	}
    }
    
    private BorderPane createImgView(Image img) {
    	BorderPane ret = new BorderPane();
    	
    	TabPane tabPane = new TabPane();
    	tabPane.setTabClosingPolicy(TabClosingPolicy.UNAVAILABLE);
    	Tab tabImg = new Tab("Image");
    	Tab tabMeta = new Tab("Meta");
    	
    	// image and caption
    	VBox imgVBox = new VBox();
    	imgVBox.setAlignment(Pos.CENTER);
	     
    	ImageView imageView = new ImageView();
    	imageView.setPreserveRatio(true);
    	imageView.setFitWidth(1024);
    	imageView.setImage(new javafx.scene.image.Image(img.imgfile.toUri().toString()));

	    Label captionView = new Label();
	    captionView.setText(img.caption);
	    captionView.setWrapText(true);
	    
	    ImageViewPane imageViewPane = new ImageViewPane(imageView);
	    imageViewPane.setMinWidth(126);
	    
	    VBox.setVgrow(captionView, Priority.ALWAYS);
	    imgVBox.getChildren().setAll(imageViewPane, captionView);
    	tabImg.setContent(imgVBox);
	    
	    // meta data and keyphrases
    	VBox metaVBox = new VBox();
    	metaVBox.setAlignment(Pos.TOP_LEFT);
    	metaVBox.setSpacing(10);
    	
	    String metaContent;
		try {
			ListView<String> keywordList = new ListView<String>();
			ObservableList<String> items = FXCollections.observableArrayList ();
			for (String kp : img.keyphrases)
				items.add(kp);
			keywordList.setItems(items);
			int ROW_HEIGHT = 24;
			keywordList.setPrefHeight(items.size()* ROW_HEIGHT + 2);
			
			metaContent = readFile(img.metafile.toString(), Charset.defaultCharset());
			JSONObject jobj = new JSONObject(metaContent);
			final TextArea metaText = new TextArea(jobj.toString(2));
			metaText.setWrapText(true);
			metaText.setEditable(false);
			
			metaVBox.getChildren().addAll(new Label("Keyphrases in Caption:"), keywordList, new Label("Meta data:"), metaText);
		} catch (IOException | JSONException e) {
			e.printStackTrace();
		}
		tabMeta.setContent(metaVBox);
	    
	    tabPane.getTabs().addAll(tabImg, tabMeta);
	    ret.setCenter(tabPane);

    	return ret;
    }
    
    private static String readFile(String path, Charset encoding) throws IOException {
	  byte[] encoded = Files.readAllBytes(Paths.get(path));
	  return new String(encoded, encoding);
	}
    
    private void showAnnotation() {
    	annotationView.setVisible(true);
    	
    	annoTitle.setText("Annotate image \""+currAnno.name+"\" (Article ID: "+currAnno.id+", Section title: "+currAnno.section+")");
        	
        radioGroupMI.selectToggle(radioBtnsMI[currAnno.annotationMI+1]);
		radioGroupSC.selectToggle(radioBtnsSC[currAnno.annotationSC+3]);
        
    	this.markedTextCB.setItems(currAnno.markedText);
    	
    	this.checkboxValid.setSelected(!currAnno.validSample);
    	
    	this.imageTypeCB.getSelectionModel().select(currAnno.imageType);
    }
    
    @FXML
    void handle(ActionEvent e) {
    	if (e.getSource() == this.addBtn) {
			Object selection = this.textViewEngine.executeScript(SELECT_TEXT);
	        if (selection instanceof String) {
	        	currAnno.markedText.add((String) selection);
	    	}
	        Backup.getInstance().backupAnnotation(currAnno);
		}
		
    	else if (e.getSource() == this.removeBtn) {
			currAnno.markedText.remove(this.markedTextCB.getValue());
			Backup.getInstance().backupAnnotation(currAnno);
		}
    	
    	else if (e.getSource() == this.checkboxValid) {
    		currAnno.validSample = !this.checkboxValid.isSelected();
    		Backup.getInstance().backupAnnotation(currAnno);
    	}
    }
    
    @FXML
    void handleScroll(ScrollEvent e) {
    	if (controlPressed) {
	    	if (e.getDeltaY() > 0)
	    		this.textView.setFontScale(this.textView.getFontScale()*1.05);
	    	else
	    		this.textView.setFontScale(this.textView.getFontScale()*0.95);
	    	e.consume();
    	}
    }
}