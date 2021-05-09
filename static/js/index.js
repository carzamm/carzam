function preloader(){
    document.getElementById("loading").style.display = "none";
    document.getElementById("content").style.display = "block";
}

// get input element once dom content loads
function domLoaded() {

	// get the original input element which is hidden
	var imageName = document.getElementById('choose-file');

	// fuction to change label text
	function editLabel() {
		// gets the filepath of the user chosen file from input element
	    var filePath = imageName.value; 

	    // gets the file name from file path
	    var fileNameStart = filePath.lastIndexOf('\\');
	    filePath = filePath.substr(fileNameStart + 1);

	    // truncate the file name if too long
	   	if (filePath.length > 13) {
	   		filePath = filePath.substring(0, 13);
	   		filePath = filePath + "...";
	    }

	    // select the label element
	    var labelToEdit = document.querySelector('label[for="choose-file"]');

	    // iterate the children of the label element and only change the text
		for(const node of labelToEdit.childNodes) {
			if(node.nodeName === '#text') {
				if (filePath !== '') {
	        		node.textContent = filePath;
	    		}
			}
		}
	}

// event listener to detect file a new file choise and edit the label text
imageName.addEventListener('change',editLabel,false);

}

// call preloader when window loads
window.onload = preloader;

// call domLoaded when dom content is loaded
document.addEventListener('DOMContentLoaded', domLoaded, false);




