function preloader(){
    document.getElementById("loading").style.display = "none";
    document.getElementById("content").style.display = "block";
}//preloader
window.onload = preloader;

console.log("test");

// get the original input element which is hidden
var imageName = document.getElementById('choose-file');
console.log(imageName);

// fuction to change label text
function editLabel() {
	console.log("test");
	// gets the filepath of the user chosen file from input element
    var filePath = imageName.value; 

    // gets the file name from file path
    var fileNameStart = filePath.lastIndexOf('\\');
    filePath = filePath.substr(fileNameStart + 1);

    // change the label text
    var newLabelText = document.querySelector('label[for="choose-file"]').childNodes[2]; /* finds the label text */
    if (filePath !== '') {
        newLabelText.textContent = filePath;
    }
}

// event listener to detect file a new file choise and edit the label text
imageName.addEventListener('change',editLabel,false);