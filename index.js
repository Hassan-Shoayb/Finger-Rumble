let mobilenet;
let model;
const webcam = new Webcam(document.getElementById('wc'));
const dataset = new RPSDataset();
var rockSamples=0, paperSamples=0, scissorsSamples=0, spockSamples=0, lizardSamples=0;
let isPredicting = false;

async function loadMobilenet() {
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

async function train() {
    const metrics = ['loss','acc'];
	const container = { name: 'Model Training', styles: { height: '640px' } };
	const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);  
    
    dataset.ys = null;
    dataset.encodeLabels(5);
    model = tf.sequential({
    layers: [
        
      // YOUR CODE HERE
    tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
    tf.layers.dense({ units: 100, activation: 'relu'}),
    tf.layers.dense({ units: 5, activation: 'softmax'})
    ]
});
    
  const optimizer = tf.train.adam(0.0001); // YOUR CODE HERE
    
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
  let loss = 0;
  model.fit(dataset.xs, dataset.ys, {
    epochs: 10,
    callbacks: fitCallbacks
   });
}


function handleButton(elem){
	switch(elem.id){
		case "0":
			rockSamples++;
			document.getElementById("rocksamples").innerText = "Rock Samples:" + rockSamples;
			break;
		case "1":
			paperSamples++;
			document.getElementById("papersamples").innerText = "Paper Samples:" + paperSamples;
			break;
		case "2":
			scissorsSamples++;
			document.getElementById("scissorssamples").innerText = "Scissors Samples:" + scissorsSamples;
			break;  
		case "3":
			spockSamples++;
			document.getElementById("spocksamples").innerText = "Spock Samples:" + spockSamples;
			break;
            
        case "4":
			lizardSamples++;
			document.getElementById("lizardsamples").innerText = "Lizard Samples:" + lizardSamples;
			break;  
            
	}
	label = parseInt(elem.id);
	const img = webcam.capture();
	dataset.addExample(mobilenet.predict(img), label);

}

async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
    var predictionText = "";
    switch(classId){
		case 0:
			predictionText = "I see Rock";
			break;
		case 1:
			predictionText = "I see Paper";
			break;
		case 2:
			predictionText = "I see Scissors";
			break;
		case 3:
			predictionText = "I see Spock";
			break;
            
        // Add a case for lizard samples.
        // HINT: Look at the previous cases.
            
        // YOUR CODE HERE 
        case 4:
			predictionText = "I see Lizard";
			break;
	
            
	}
	document.getElementById("prediction").innerText = predictionText;
			
    
    predictedClass.dispose();
    await tf.nextFrame();
  }
}


function doTraining(){
	train();
	alert("Training Done!")
}

function startPredicting(){
	isPredicting = true;
	predict();
}

function stopPredicting(){
	isPredicting = false;
	predict();
}


function saveModel(){
    model.save('downloads://my_model');
}


async function init(){
    tfvis.show.modelSummary({name: 'Model Architecture'}, model);
	await webcam.setup();
	mobilenet = await loadMobilenet();
	tf.tidy(() => mobilenet.predict(webcam.capture()));
    
		
}


init();