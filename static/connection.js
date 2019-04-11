
function response_handler(res){
	if(res.class=="A")
		console.log(":(");
	else
		console.log(":)");
}


function ShowCam() {
    Webcam.set({
        width: 320,
        height: 240,
        image_format: 'jpeg',
        jpeg_quality: 100
    });
    Webcam.attach('#my_camera');
}
window.onload= function (){
    ShowCam();
    repeat(2000);

};

var makeASound = function(sound_src){
    //TODO
};

function uploadImage(src) {
    var formData = new FormData();
    formData.append("file", src);
    var xmlhttp = new XMLHttpRequest();
    xmlhttp.open("POST", "/process");
    // check when state changes, 
    xmlhttp.onreadystatechange = function() {

	    if(xmlhttp.readyState == 4 && xmlhttp.status == 200) {
	       var res = JSON.parse(xmlhttp.responseText);
	       response_handler(res);

	    }
	}
	    
	xmlhttp.send(formData);  
}



function snap() {
    Webcam.snap( function(data_uri) {
        // display results in page
        // document.getElementById('results').innerHTML = 
        // '<img id="image" src="'+data_uri+'"/>';
        uploadImage(data_uri);

      } );      
}




$('#upload').on('click',uploadImage);


var repeat = function(time){
    setInterval(snap, time)
};