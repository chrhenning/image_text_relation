package annotation;

public class LabelDescription {
	// Inclusion: How well described is the image by the text?
	public static final String[] TextIncludesImage = {
			"The text describes the opposite of what is depicted in the image.", // -2
			"One of the main story characteristics of the text is incorrectly depicted by the image. (e.g. a male protagonist is depicted by a female person)\nor\nThe actual content of the text has no relation to the image, but the text refers to some details which are either depicted as described or at least depicted similarly in the image. (e.g. A text about nutrient-rich soil in forests has an image that depicts an indoor plant)",
			"The contents of the text and the image do not overlap. Not even a contextual relationship can be built.",
			"The text only describes a part of the image.\nor\nThe text describes a storyline which is usually associated with the image due to background knowledge.",
			"The text sketches the image but on an abstract level such that many different pictures would be similarly well described by the text.",
			"The text describes a process/story, where an excerpt is depicted in the image. Particularly, the text references to details in the image.",
			"The text describes exclusively the image. The content description is complete and rich in detail and all outstanding characteristics are mentioned. (The text alone should be sufficient to be able to imagine the depicted scenery without prior knowledge of the image.)"
			};
	
	// Inclusion: How well described is the text by the image?
	public static final String[] ImageIncludesText = {
			"The image shows the opposite of what the text describes.", // -2
			"The depicted scenery fits to the text, but was incorrectly interpreted.\nor\nA critical detail of the image leads an opposite interpretation of text and image. (e.g. a campaign event of the Republic party is shown, whereas  the text reports about a campaign event of the Democratic party)\nor\nKeywords from the text can be identified as actions or objects in the image, but they appear in a whole different context.",
			"The contents of the text and the image do not overlap. Not even a contextual relationship can be built.",
			"The image concentrates on a detail of the text.\nor\nThe content of the image is not explicitly mentioned in the text. However, due to background knowledge the image can be brought into context.",
			"The image can be viewed as a summary of the text content, but only by neglecting many of the details. (e.g. photo of the conflict area, whereof the current news story is reporting about) ",
			"The image depicts, subjectively seen, more important characteristics than the text describes. Though, part of the image is described exhaustively. ",
			"The image depicts solely situations/actions/characteristics that are mentioned in the text. Nothing is depicted which can not be concluded from the text."
	};
}
