package application;
	
import de.codecentric.centerdevice.javafxsvg.SvgImageLoaderFactory;
import javafx.application.Application;
import javafx.stage.Stage;
import javafx.scene.Scene;


public class Main extends Application {
	
	private static AppView view;
	
	@Override
	public void start(Stage primaryStage) {
		try {
			view = new AppView(primaryStage); 
			primaryStage.setMaximized(true);
			primaryStage.setTitle("wikiAnnotator");
			Scene scene = new Scene(view,800,800);
			scene.getStylesheets().add(getClass().getResource("application.css").toExternalForm());
			primaryStage.setScene(scene);
			view.createGUI();
			view.setup();
			//System.out.println( System.getProperties());
			primaryStage.show();
		} catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	@Override
	public void stop(){
	    view.programClosed();
	}
	
	public static void main(String[] args) {
		SvgImageLoaderFactory.install();
		
		launch(args);
	}
}