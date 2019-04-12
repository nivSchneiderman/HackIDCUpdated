var countAll=0;
var classA=0;
var classB=0;
var classC=0;
var classD=0;


var inARow=0;


function response_handler(res){
	console.log(res.class);
    countAll++;
    switch(res.class) {
      case "A":
       classA++;
	   inARow=0;
        break;
      case "B":
         classB++;
		 inARow++;
        break;
        case "C":
         classC++;
		 inARow++; 
        break;
        case "D":
         classD++;
      default:
        // code block
    }

    $('#gauge2 .gauge-arrow').trigger('updateGauge', ((classB+classC)/countAll)*100);
    $('#gauge2 .gauge-arrow').cmGauge();
    $('#gauge1 .gauge-arrow').trigger('updateGauge', ((classB+classC)/countAll)*100);
    $('#gauge1 .gauge-arrow').cmGauge();
//    var randomNum = Math.floor((Math.random() * 100));
//    $('#gauge1 .gauge-arrow').trigger('updateGauge', randomNum);
//     $('#gauge1 .gauge-arrow').cmGauge();
	 
	 if(inARow>5){
		 makeASound();
	 }
    
}




function ShowCam() {
    Webcam.set({
        width: 320,
        height: 240,
        image_format: 'jpeg',
        jpeg_quality: 90
    });
    Webcam.attach('#my_camera');
}
window.onload= function (){
    ShowCam();
    repeat(1000);

};

function makeASound() {
    if(document.getElementById('soundOn').checked){
        var audio = new Audio('/static/sounds/beep1.wav');
         audio.play();
    } 
};

$("#videoOn").on('click', function(){
   if( $(this).is(':checked') ){

   }else{
    $(this).fadeout(300);
   } 
});

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