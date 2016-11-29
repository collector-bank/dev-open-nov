var config = {
	isSecure: true,
	host: "xxx",
	port: 443,
	prefix: "/"
};


require.config({baseUrl:"https://xxx:443/resources"});

var qlikapp;

require( ["js/qlik"], function ( qlik ) {
    
    getQlikApp = function() {
        return qlik.openApp('IdForSenseApp', config)
    };
        
    
    qlikApp = getQlikApp();
    qlikApp.getObject('QV01', 'IdForSenseObject1');
    qlikApp.getObject('QV02', 'IdForSenseObject2');
    
    
    qlikApp.visualization.create('scatterplot',["Go.Name", "=sum(Go.W)", "=sum(Go.SV%)", "=Sum(Go.HighS)"],
        {title:'Goaltender',
         "color": {
			"auto": false, 
			"mode": "byMeasure",
			"measureScheme": "dg" 
		  }
        }).then(function(bar){bar.show('QV03');
    });
    
    
    
    clearAll = function() {
        qlikApp.clearAll();
        $('#menu .dropdown button').html('Team <span class="caret"></span>'); 
    };
    
    back = function() {
        qlikApp.back();
    };
    
    forward = function() {
        qlikApp.forward();
    };
    
    selectValues = function(field, values) {
        qlikApp.field(field).selectValues([values], false, false);
    };
    
	qlik.setOnError( function ( error ) {
		$('#popupText').append(error.message +"<br>");
        popup();
	} );
 
    
    
    
    
    
    //callbacks
    function showData(reply, qlikApp){
        $('#menu .dropdown ul').empty()  
        $.each(reply.qListObject.qDataPages[0].qMatrix, function(key, value) {  
            if (typeof value[0].qText !== 'undefined') {  
                $('#menu .dropdown ul').append('<li><a data-select="'+ value[0].qText+'" href="#">'+ value[0].qText+'</a></li>');  
            }  
        });  
    };
    
    qlikApp.createList({
		"qFrequencyMode": "V",
		"qDef": {
				"qFieldDefs": ["End Team"]
		},
		"qExpressions": [],
		"qInitialDataFetch": [
				{
						"qHeight": 50,
						"qWidth": 1
				}
		],
		"qLibraryId": null
	},showData);
    
    //jquery
    $('body').on( "click", "[data-select]", function() {  
        var value = $(this).data('select');  
        selectValues('[End Team]', value);
        $('#menu .dropdown button').html(value + ' <span class="caret"></span>');  
  });  
    
} );


