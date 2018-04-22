package application;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Optional;
import java.util.concurrent.TimeUnit;

import javafx.application.Platform;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.Alert;
import javafx.scene.control.Alert.AlertType;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonType;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.Menu;
import javafx.scene.control.MenuBar;
import javafx.scene.control.MenuItem;
import javafx.scene.control.Pagination;
import javafx.scene.control.RadioButton;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.SplitPane;
import javafx.scene.control.TextField;
import javafx.scene.control.Toggle;
import javafx.scene.control.ToggleGroup;
import javafx.scene.control.Tooltip;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyEvent;
import javafx.scene.input.ScrollEvent;
import javafx.scene.layout.AnchorPane;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.scene.web.WebEngine;
import javafx.scene.web.WebView;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import util.DatasetHandler;
import util.ImageViewPane;
import util.Sample;

public class AppView implements EventHandler<ActionEvent> {
	private Stage primaryStage;
	
	private ImageView imageView;
	private Label captionView;
	private WebEngine webViewEngine;
	
	private SplitPane splitPane1;
	
	private Pagination pagination;
	
	private DatasetHandler dataset;
	
	// text includes image
	private ToggleGroup radioGroupMI;
	private RadioButton[] radioBtnsMI = {null,null,null,null,null,null,null,null,null};
	
	// image includes text
	private ToggleGroup radioGroupSC;
    private RadioButton[] radioBtnsSC = {null,null,null,null,null,null};
	
	private Sample currentSample;
	
	private Button saveBtn;
	private Button saveAsBtn;
	
	private File saveFile = null;
	
	private TextField pageSelector;
	
	public ComboBox<String> markedTextCB;
	private Button addBtn;
	private Button removeBtn;
	
	public boolean controlPressed;
	
	private MenuItem menuItemClose;
	//private MenuItem menuItemOpen;
	
	private ComboBox<String> imageTypeCB;
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
	
	public AppView(BorderPane root, Stage stage) {
		this.primaryStage = stage;
		this.dataset = new DatasetHandler();
		
		this.createGUI(root, this.primaryStage);
	}
	
	private void createGUI(BorderPane root, Stage stage) {
		root.setStyle("-fx-background: #e6ffff;");
		
		/*
		 * Menu bar
		 */
		MenuBar menuBar = new MenuBar();
        Menu menuFile = new Menu("File");
        //this.menuItemOpen = new MenuItem("Open Dataset");
        this.menuItemClose = new MenuItem("Close");
        
        //this.menuItemOpen.setOnAction(this);
        this.menuItemClose.setOnAction(this);
        
        menuFile.getItems().addAll(this.menuItemClose);
        //menuFile.getItems().addAll(this.menuItemOpen, new SeparatorMenuItem(), this.menuItemClose);
        menuBar.getMenus().addAll(menuFile);
        root.setTop(menuBar);
		
		/*
		 * define split pane
		 */
		this.splitPane1 = new SplitPane();
		
	    WebView browser = new WebView();
        this.webViewEngine = browser.getEngine();
        browser.setOnScroll(new EventHandler<ScrollEvent>() {
			@Override
			public void handle(ScrollEvent e) {
				if (controlPressed) {
			    	if (e.getDeltaY() > 0)
			    		browser.setFontScale(browser.getFontScale()*1.05);
			    	else
			    		browser.setFontScale(browser.getFontScale()*0.95);
			    	e.consume();
		    	}
			}	
        });

        ScrollPane textView = new ScrollPane();
        textView.setContent(browser);
        textView.setFitToHeight(true);
        textView.setFitToWidth(true);

        browser.prefWidthProperty().bind(textView.widthProperty());
        browser.prefHeightProperty().bind(textView.heightProperty());	    
	    
	    VBox imgView = new VBox();
	    imgView.setAlignment(Pos.CENTER);
	    
	    this.imageView = new ImageView();
	    this.imageView.setPreserveRatio(true);
	    this.imageView.setFitWidth(1024);

	    this.captionView = new Label();
	    this.captionView.setWrapText(true);
	    
	    ImageViewPane imageViewPane = new ImageViewPane(this.imageView);
	    imageViewPane.setMinWidth(126);
	    imgView.getChildren().setAll(imageViewPane, captionView);

	    this.splitPane1.getItems().addAll(textView, imgView);
	    
	    /*
	     * Define pagination which contains the split pane as single page
	     */
	    this.pagination = new Pagination(this.dataset.getNumSamples(), 0);
	    this.pagination.setPageFactory((Integer pageIndex) -> {
            if (pageIndex >= this.dataset.getNumSamples()) {
            	return null;
            } else {
            	this.setContent(this.dataset.getSample(pageIndex));
                return this.splitPane1;
            }
        });
 
        AnchorPane page = new AnchorPane();
        AnchorPane.setTopAnchor(this.pagination, 10.0);
        AnchorPane.setRightAnchor(this.pagination, 10.0);
        AnchorPane.setBottomAnchor(this.pagination, 10.0);
        AnchorPane.setLeftAnchor(this.pagination, 10.0);
        page.getChildren().addAll(this.pagination);
        
        /*
         * Define annotation
         */
        VBox bottomView = new VBox();
        bottomView.setPadding(new Insets(15, 12, 15, 12));
        bottomView.setSpacing(10);
        
        HBox annotationTopView = new HBox();
        annotationTopView.setSpacing(10);
        annotationTopView.setAlignment(Pos.BOTTOM_LEFT);
        
        Label annotationHeading = new Label("Annotate above sample:");
        annotationHeading.setFont(Font.font("Cambria", FontWeight.BOLD, 20));
        
        HBox pageSelectionView = new HBox();
        pageSelectionView.setAlignment(Pos.CENTER_RIGHT);
        pageSelectionView.setSpacing(10);
        
        this.pageSelector = new TextField();
        this.pageSelector.setPrefWidth(80);
		// force the field to be numeric only
        this.pageSelector.textProperty().addListener(new ChangeListener<String>() {
			@Override
			public void changed(ObservableValue<? extends String> observable, String oldValue, String newValue) {
				if (!newValue.matches("\\d*")) {
					pageSelector.setText(newValue.replaceAll("[^\\d]", ""));
				}
			}
		});
		// handle page change if enter was hit
        this.pageSelector.setOnKeyPressed(new EventHandler<KeyEvent>() {
		    @Override
			public void handle(KeyEvent keyEvent) {
				if (keyEvent.getCode() == KeyCode.ENTER) {
					int pn = Integer.parseInt(pageSelector.getText()) - 1;
					if (pn < 0 || pn >= pagination.getPageCount()) {
						// signalize error by drawing textfield red
						Thread thread = new Thread() {
							public void run() {
								pageSelector.setStyle("-fx-text-box-border: red ;-fx-focus-color: red ;");
								try {
									TimeUnit.SECONDS.sleep(1);
								} catch (InterruptedException e) {
									// TODO Auto-generated catch block
									e.printStackTrace();
								}
								pageSelector.setStyle("");
							}
						};
						thread.start();
					} else { // change page
						pagination.setCurrentPageIndex(pn);
					}

				}
			}
		});		
        
        pageSelectionView.getChildren().addAll(new Label("Go to page:"), this.pageSelector);
        
        HBox.setHgrow(pageSelectionView, Priority.ALWAYS);
        annotationTopView.getChildren().addAll(annotationHeading, pageSelectionView);
        
        // mutual information
        this.radioGroupMI = new ToggleGroup();
        
        HBox radioBtnsHBoxMI = new HBox();
        radioBtnsHBoxMI.setAlignment(Pos.CENTER);
        radioBtnsHBoxMI.setPadding(new Insets(15, 12, 15, 12));
        radioBtnsHBoxMI.setSpacing(10);  

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
					int newAnno = (int) radioGroupMI.getSelectedToggle().getUserData();
					dataset.changeAnnotationMI(currentSample, newAnno);
				}
			}
		});
        
        // semantic correlation
        this.radioGroupSC = new ToggleGroup();
        
        HBox radioBtnsHBoxSC = new HBox();
        radioBtnsHBoxSC.setAlignment(Pos.CENTER);
        radioBtnsHBoxSC.setPadding(new Insets(15, 12, 15, 12));
        radioBtnsHBoxSC.setSpacing(10);  

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
					int newAnno = (int) radioGroupSC.getSelectedToggle().getUserData();
					dataset.changeAnnotationSC(currentSample, newAnno);
				}
			}
		});
        
        // marking texts
        HBox markedTextHBox = new HBox();
        markedTextHBox.setAlignment(Pos.CENTER);
        markedTextHBox.setPadding(new Insets(15, 12, 15, 12));
        markedTextHBox.setSpacing(10);  
        
        markedTextCB = new ComboBox<>();
        this.addBtn = new Button("Add");
        this.addBtn.setOnAction(this);

        this.removeBtn = new Button("Remove");
        this.removeBtn.setOnAction(this);
        
        markedTextHBox.getChildren().addAll(markedTextCB, addBtn, removeBtn);
        
        // image types
        HBox typeHBox = new HBox();
        typeHBox.setPadding(new Insets(15, 12, 15, 12));
        typeHBox.setSpacing(10); 
        
        ObservableList<String> imageTypes = FXCollections.observableArrayList(imgTypes);
        this.imageTypeCB = new ComboBox<>();
        this.imageTypeCB.setItems(imageTypes);
        this.imageTypeCB.setOnAction((event) -> {
            String selectedType = this.imageTypeCB.getSelectionModel().getSelectedItem();
            this.currentSample.imageType = selectedType;
            this.dataset.backupSample(this.currentSample);
        });
        
        // save buttons        
        HBox saveBtnView = new HBox();
        saveBtnView.setSpacing(10);
        saveBtnView.setAlignment(Pos.BOTTOM_RIGHT);
        
        this.saveBtn = new Button("Save");
        this.saveBtn.setOnAction(this);
        this.saveBtn.setDisable(true);
        
        this.saveAsBtn = new Button("Save As");
        this.saveAsBtn.setOnAction(this);
        //this.saveAsBtn.setDefaultButton(true);
        
        saveBtnView.getChildren().addAll(this.saveBtn, this.saveAsBtn);
        
        Label labelMI = new Label("Mutual Information (CMI) of Text and Image");
        labelMI.setFont(new Font("Cambria", 20));
        Label labelSC = new Label("Semantic Correlation (SC) of Text and Image");
        labelSC.setFont(new Font("Cambria", 20));
        Label labelMarked = new Label("Add marked text snippets, which are highly correlated to the image.");
        labelMarked.setFont(new Font("Cambria", 20));
        Label labelType = new Label("Type of image: ");
        labelType.setFont(new Font("Cambria", 20));
        typeHBox.getChildren().addAll(labelType, this.imageTypeCB);
        
        VBox.setVgrow(annotationTopView, Priority.ALWAYS);
        VBox.setVgrow(saveBtnView, Priority.ALWAYS); 
        bottomView.getChildren().addAll(annotationTopView, labelMI, radioBtnsHBoxMI, labelSC, radioBtnsHBoxSC, labelMarked, markedTextHBox, typeHBox, saveBtnView);
        
        root.setCenter(page);
        root.setBottom(bottomView);
        
		primaryStage.getScene().setOnKeyPressed(new EventHandler<KeyEvent>() {
			@Override
			public void handle(KeyEvent e) {
				if (e.getCode() == KeyCode.SPACE || e.getCode() == KeyCode.RIGHT) {
					if (pagination.getCurrentPageIndex() < pagination.getPageCount()-1) {
						pagination.setCurrentPageIndex(pagination.getCurrentPageIndex()+1);
					}
				} else if (e.getCode() == KeyCode.LEFT) {
					if (pagination.getCurrentPageIndex() > 0) {
						pagination.setCurrentPageIndex(pagination.getCurrentPageIndex()-1);
					}
				} else if (KeyCode.CONTROL == e.getCode())
    				controlPressed = true;
			}
		});
	    	
        primaryStage.getScene().setOnKeyReleased(new EventHandler<KeyEvent>() {
            @Override
            public void handle(KeyEvent event) {
            	if (KeyCode.CONTROL == event.getCode())
    				controlPressed = false;
            }
        });
	}

	/**
	 * Update the content with a new sample
	 * 
	 * @param sample
	 */
	private void setContent(Sample sample) {
		this.currentSample = sample;
		
		this.webViewEngine.loadContent(DatasetHandler.readText(sample.text));
		this.imageView.setImage(new Image(sample.image.toUri().toString()));
		this.captionView.setText(DatasetHandler.readCaption(sample.caption));

		radioGroupMI.selectToggle(radioBtnsMI[sample.annotationMI+1]);
		radioGroupSC.selectToggle(radioBtnsSC[sample.annotationSC+3]);
		
		this.markedTextCB.setItems(sample.markedText);
		
		this.imageTypeCB.getSelectionModel().select(sample.imageType);
		
		System.out.println(sample.name);
	}
	
	/**
	 * Should be called if stage is closing
	 */
	public void programClosed() {
		// if has not changed yet
		// ...
		
		this.dataset.closeDB();
	}

	@Override
	public void handle(ActionEvent e) {
		
		if (e.getSource() == menuItemClose) {
			Platform.exit();
		} 
		        
		if (e.getSource() == this.saveBtn || e.getSource() == this.saveAsBtn) {
			if (e.getSource() == this.saveAsBtn) {
				FileChooser fileChooser = new FileChooser();
				fileChooser.setTitle("Save annotations in ...");
				fileChooser.setInitialFileName("bbcAnnotations");
				fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("JSON Lines (*.jsonl)", "*.jsonl"));
				File file = fileChooser.showSaveDialog(this.primaryStage);
                if (file != null) {
                    this.saveFile = new File(file.getAbsolutePath().endsWith(".jsonl") ? file.getAbsolutePath() : file.getAbsolutePath() + ".jsonl");
                    
                    if (this.saveFile.exists()) {
	                    Alert alert = new Alert(AlertType.CONFIRMATION);
	                    alert.setTitle("File already exists");
	                    alert.setHeaderText("This operation will overwrite the file " + this.saveFile.getName());
	                    alert.setContentText("Are you ok with this?");
	
	                    Optional<ButtonType> result = alert.showAndWait();
	                    if (result.get() != ButtonType.OK){
	                        this.saveFile = null;
	                        return;
	                    }
                    }
                    
                    System.out.println(saveFile);
                    this.saveBtn.setDisable(false);
                } else {
                	return;
                }
			}
			
			assert(this.saveFile != null);
			try {
				this.dataset.saveAnnotations(this.saveFile);
			} catch (FileNotFoundException e1) {
				Alert alert = new Alert(AlertType.ERROR);
				alert.setTitle("Could not save annotations");
				alert.setHeaderText("An error has occured while trying to save the annotations.");
				alert.setContentText(e1.getMessage());
				alert.showAndWait();
				
				e1.printStackTrace();
			}
		}
		
		if (e.getSource() == this.addBtn) {
			Object selection = this.webViewEngine.executeScript(SELECT_TEXT);
	        if (selection instanceof String) {
	        	currentSample.markedText.add((String) selection);
	    	}
	        this.dataset.backupSample(currentSample);
		}
		
		if (e.getSource() == this.removeBtn) {
			currentSample.markedText.remove(this.markedTextCB.getValue());
			this.dataset.backupSample(currentSample);
		}
	}
}


