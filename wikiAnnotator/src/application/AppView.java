package application;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Optional;
import java.util.concurrent.TimeUnit;

import data.Backup;
import data.DatasetHandler;
import javafx.application.Platform;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.control.Alert;
import javafx.scene.control.ButtonType;
import javafx.scene.control.MenuItem;
import javafx.scene.control.Pagination;
import javafx.scene.control.TextField;
import javafx.scene.control.Alert.AlertType;
import javafx.scene.control.Button;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyEvent;
import javafx.scene.layout.BorderPane;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

public class AppView extends BorderPane {
	private Stage stage;
	private DatasetHandler dsHandler = null;
    
    @FXML
    private MenuItem menuItemOpenDS;

    @FXML
    private MenuItem menuItemClose;
    
    @FXML
    private Pagination samplePagination;
    
    @FXML
    private TextField pageSelector;
    
    @FXML
    private Button saveAsBtn;
    
    @FXML
    private Button saveBtn;
    
    private File saveFile = null;
	
	public AppView(Stage stage) {
		try {
            FXMLLoader loader = new FXMLLoader(getClass().getResource("AppView.fxml"));
            loader.setController(this);
            loader.setRoot(this);
            loader.load();
        } catch (IOException exc) {
        	exc.printStackTrace();
        }
		
		this.stage = stage;
		this.dsHandler = DatasetHandler.getInstance();
	}
	
	public void createGUI() {
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
					if (pn < 0 || pn >= samplePagination.getPageCount()) {
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
						samplePagination.setCurrentPageIndex(pn);
					}

				}
			}
		});		
	}
	
	// things that have to be done after the GUI was initialized
	public void setup() {
		String dsPath = Backup.getInstance().getDataSetPath();
		if (dsPath != null) {
			this.changeDataSet( new File(dsPath));				
		}
		
		this.stage.getScene().setOnKeyPressed(new EventHandler<KeyEvent>() {
			@Override
			public void handle(KeyEvent e) {
				if (e.getCode() == KeyCode.SPACE || e.getCode() == KeyCode.RIGHT) {
					if (samplePagination.getCurrentPageIndex() < samplePagination.getPageCount()-1) {
						samplePagination.setCurrentPageIndex(samplePagination.getCurrentPageIndex()+1);
					}
				} else if (e.getCode() == KeyCode.LEFT) {
					if (samplePagination.getCurrentPageIndex() > 0) {
						samplePagination.setCurrentPageIndex(samplePagination.getCurrentPageIndex()-1);
					}
				}
			}
		});
	}
	
	/**
	 * Should be called if stage is closing
	 */
	public void programClosed() {
		//this.dataset.closeDB();
	}
	
	private void changeDataSet(File file) {
		if (file != null && file.exists()) {
			this.dsHandler.loadDataset(file);

			samplePagination.setPageCount(this.dsHandler.getNumSamples());
			this.samplePagination.setPageFactory((Integer pageIndex) -> {
	            if (pageIndex >= this.dsHandler.getNumSamples()) {
	            	return null;
	            } else {
	            	SampleView sv = new SampleView(this.stage);
	            	sv.setSample(pageIndex);
	                return sv;
	            }
	        });
		}
	}
	
	@FXML
    void handle(ActionEvent e) {
		
		if (e.getSource() == menuItemClose) {
			Platform.exit();
		} 
		
		else if (e.getSource() == menuItemOpenDS) {
			FileChooser fileChooser = new FileChooser();
			fileChooser.setTitle("Read data set from ...");
			fileChooser.setInitialFileName("articles.json");
			//fileChooser.setInitialDirectory(new File(System.getProperty("user.home"));
			fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("JSON Lines (*.jsonl)", "*.jsonl"));
			File file = fileChooser.showOpenDialog(this.stage);
			String currPath = Backup.getInstance().getDataSetPath();
			if (currPath != null && !file.equals(new File(currPath))) {
				Alert alert = new Alert(AlertType.CONFIRMATION);
                alert.setTitle("Delete current backup");
                alert.setHeaderText("This operation will overwrite the annotations for data set " + currPath);
                alert.setContentText("Are you ok with this?");

                Optional<ButtonType> result = alert.showAndWait();
                if (result.get() == ButtonType.OK){
                	Backup.getInstance().clearDB();
                } else {
                	return;
                }
			}
			this.changeDataSet(file);
		}
		
		else if (e.getSource() == this.saveBtn || e.getSource() == this.saveAsBtn) {
			if (e.getSource() == this.saveAsBtn) {
				FileChooser fileChooser = new FileChooser();
				fileChooser.setTitle("Save annotations in ...");
				fileChooser.setInitialFileName("wikiAnnotations");
				fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("JSON Lines (*.jsonl)", "*.jsonl"));
				File file = fileChooser.showSaveDialog(this.stage);
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
				this.dsHandler.saveAnnotations(this.saveFile);
			} catch (FileNotFoundException e1) {
				Alert alert = new Alert(AlertType.ERROR);
				alert.setTitle("Could not save annotations");
				alert.setHeaderText("An error has occured while trying to save the annotations.");
				alert.setContentText(e1.getMessage());
				alert.showAndWait();
				
				e1.printStackTrace();
			}
		}
	}
}

