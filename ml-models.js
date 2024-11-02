document.getElementById("file-input").addEventListener("change", handleFileUpload);
document.getElementById("model-select").addEventListener("change", resetModel);

let xTrain = [];
let yTrain = [];
let xToPredict = [];
let model;

//--------------------Decision Tree--------------------
let headers = [];
let training = "";
let arrTraining = [];
//-----------------------------------------------------

//-------------------Neuronal Network------------------
let arrNR = [];
//-----------------------------------------------------

//-------------------Neuronal Network------------------
let clusters; //number
let trainingData; // array of numbers
let iterations; //number

//-----------------------------------------------------


// Función para limpiar datos al cambiar de modelo
function resetModel() {
    xTrain = [];
    yTrain = [];
    xToPredict = [];
    model = null;
    document.getElementById("logRS").innerHTML = "";
    if (window.myChart) {
        window.myChart.destroy();
    }

    // Muestra el selector de grado si es una regresión polinomial
    const modelType = document.getElementById("model-select").value;
    document.getElementById("degree-select-group").style.display = modelType === "polynomial-regression" ? "block" : "none";
    document.getElementById("percent-select-group").style.display = modelType === "linear-regression" ? "block" : "none";
    document.getElementById("data-select-group").style.display = modelType === "decision-tree" ? "block" : "none";
}


function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            const csvData = e.target.result;
            parseCSVData(csvData);
        };
        reader.readAsText(file);
    }
}

// Parseo del CSV
function parseCSVData(data) {
    const modelType = document.getElementById("model-select").value;
    console.log("modelType: ", modelType);
    const lines = data.trim().split('\n').slice(1);
    xTrain = [];
    yTrain = [];
    xToPredict = [];

    if (modelType == "linear-regression") {
        lines.forEach(line => {
            const [x, y] = line.split(';').map(parseFloat);
            xTrain.push(x);
            yTrain.push(y);
        });
        console.log("XTrain:", xTrain);
        console.log("YTrain:", yTrain);
    } else if (modelType == "polynomial-regression") {
        lines.forEach(line => {
            const [x, y, xPred] = line.split(';').map(parseFloat);
            xTrain.push(x);
            yTrain.push(y);
            xToPredict.push(xPred);
        });
        console.log("xToPredict:", xToPredict);
        console.log("XTrain:", xTrain);
        console.log("YTrain:", yTrain);

    } else if (modelType === "decision-tree") {
        // Leer encabezado y datos para el Árbol de Decisión
        headers = lines[0].split(','); // Cabeceras de atributos
        const trainData = [];

        // Incluir todas las líneas (desde la primera línea de datos hasta la última) en `trainData`
        lines.slice(1).forEach(line => {
            const attributes = line.split(',').map(item => item.trim());
            trainData.push(attributes);
        });

        // Definir `xTrain` como los datos de entrenamiento completos junto con las cabeceras
        xTrain = [headers, ...trainData];

        console.log("Headers:", headers);
        console.log("Training Data:", xTrain);
    } else if (modelType == "neuronal-network") {
        arrNR = lines[0].split(',').map(Number); // Convierte cada valor a entero

        console.log("arrNR:", arrNR);
    } else if (modelType === "k-means-linear") {
        lines.forEach(line => {
            const [numeroClusters, entrenamiento, numeroIteraciones] = line.split(';');

            // Convertimos `NumeroClusters` y `NumeroIteraciones` a enteros
            clusters = parseInt(numeroClusters, 10);
            iterations = parseInt(numeroIteraciones, 10);

            // Convertimos `Entrenamiento` en un arreglo de números
            trainingData = entrenamiento.split(',').map(Number);

            console.log("Numero de Clusters:", clusters);
            console.log("Entrenamiento:", trainingData);
            console.log("Numero de Iteraciones:", iterations);
        });
    }

}

// Clase de Regresión Lineal
class LinearRegression {
    constructor() {
        this.m = 0;
        this.b = 0;
        this.isFit = false;
    }

    fit(xTrain, yTrain) {
        let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;

        if (xTrain.length !== yTrain.length) {
            throw new Error('Los parámetros para entrenar no tienen la misma longitud!');
        }

        for (let i = 0; i < xTrain.length; i++) {
            sumX += xTrain[i];
            sumY += yTrain[i];
            sumXY += xTrain[i] * yTrain[i];
            sumXX += xTrain[i] * xTrain[i];
        }

        const n = xTrain.length;
        this.m = (n * sumXY - sumX * sumY) / (n * sumXX - Math.pow(sumX, 2));
        this.b = (sumY * sumXX - sumX * sumXY) / (n * sumXX - Math.pow(sumX, 2));
        this.isFit = true;
    }

    predict(xTest) {
        return this.isFit ? xTest.map(x => this.m * x + this.b) : [];
    }
}

// Clase de Regresión Polinomial
class PolynomialRegression {
    //PolynomialRegression code
    constructor() {
        this.coefficients = [];
        this.isFit = false;
    }

    //Method that trains the model in order to create the regression
    fit(xArray, yArray, degree) {
        //Equation matrix size based on the degree and number of elements
        let equationSize = degree + 1;
        let nElements = degree + 2;

        //Equation matrix to be solved
        let equations = new Array(equationSize);
        for (let i = 0; i < equationSize; i++) {
            equations[i] = new Array(nElements);
        }

        //Building equation matrix
        for (let i = 0; i < equationSize; i++) {
            for (let j = 0; j < nElements; j++) {
                let sum = 0;
                if (i == 0 && j == 0) {
                    sum = xArray.length;
                }
                else if (j == nElements - 1) {
                    for (let k = 0; k < xArray.length; k++) {
                        sum += Math.pow(xArray[k], i) * yArray[k];
                    }
                }
                else {
                    for (let k = 0; k < xArray.length; k++) {
                        sum += Math.pow(xArray[k], (j + i));
                    }
                }
                equations[i][j] = sum;
            }
        }

        //Staggering matrix
        for (let i = 1; i < equationSize; i++) {
            for (let j = 0; j <= i - 1; j++) {
                let factor = equations[i][j] / equations[j][j];
                for (let k = j; k < nElements; k++) {
                    equations[i][k] = equations[i][k] - factor * equations[j][k];
                }
            }
        }

        //Solving matrix
        for (let i = equationSize - 1; i > -1; i--) {
            for (let j = equationSize - 1; j > -1; j--) {
                if (i == j) {
                    equations[i][nElements - 1] = equations[i][nElements - 1] / equations[i][j];
                }
                else if (equations[i][j] != 0) {
                    equations[i][nElements - 1] -= equations[i][j] * equations[j][nElements - 1];
                }
            }
        }

        //Storing solutions
        this.solutions = new Array(equationSize);
        for (let i = 0; i < equationSize; i++) {
            this.solutions[i] = equations[i][nElements - 1];
        }

        //Setting Model as trained
        this.isFit = true;

        //Setting error
        this.calculateR2(xArray, yArray);
    }

    //Function that creates a prediction based in the regression model
    predict(xArray) {
        let yArray = [];
        //Checking if the model is already trained
        if (this.isFit) {
            //Generating the predictions based in the input and solutions
            for (let i = 0; i < xArray.length; i++) {
                let yprediction = 0;
                for (let j = 0; j < this.solutions.length; j++) {
                    yprediction += this.solutions[j] * Math.pow(xArray[i], j);
                }
                yArray.push(yprediction);
            }
        }

        //Returning Prediction
        return yArray;
    }

    //Method that stores error for the trained array
    calculateR2(xArray, yArray) {
        //Setting error array and predictions
        let errors = new Array(xArray.length);
        let prediction = this.predict(xArray);
        let sumY = 0;

        //Calculating errors
        for (let i = 0; i < xArray.length; i++) {
            sumY += yArray[i];
            errors[i] = Math.pow(yArray[i] - prediction[i], 2);
        }

        let sr = 0;
        let st = 0;
        for (let i = 0; i < xArray.length; i++) {
            sr += errors[i];
            st += Math.pow(yArray[i] - (sumY / xArray.length), 2);
        }
        let r2 = (st - sr) / st;
        this.error = r2;
    }

    getError() {
        return this.error;
    }
}

// Función para dividir datos en entrenamiento y prueba
function splitData(trainPercent) {
    const trainSize = Math.floor(xTrain.length * (trainPercent / 100));
    const xTrainSplit = xTrain.slice(0, trainSize);
    const yTrainSplit = yTrain.slice(0, trainSize);
    const xTestSplit = xTrain.slice(trainSize);
    const yTestSplit = yTrain.slice(trainSize);

    return { xTrainSplit, yTrainSplit, xTestSplit, yTestSplit };
}

// Entrenar el modelo
function trainModel() {
    const modelType = document.getElementById("model-select").value;
    const trainPercent = parseInt(document.getElementById("train-test-split").value);
    console.log("trainPercent: ", trainPercent);
    const { xTrainSplit, yTrainSplit } = splitData(trainPercent);

    if (modelType === "linear-regression") {
        model = new LinearRegression();
        model.fit(xTrainSplit, yTrainSplit);

        let yPredict = model.predict(xTrain);
        yPredict = yPredict.map(value => Number(value.toFixed(2)));
        console.log(yPredict);
        document.getElementById("logRS").innerHTML = `
            <strong>Datos de Entrenamiento:</strong><br>
            X Train: ${xTrainSplit}<br>
            Y Train: ${yTrainSplit}<br>
            Y Predict: ${yPredict}
            `;
        drawChart(xTrain, yTrain, yPredict);
    } else if (modelType === "polynomial-regression") {
        const degree = parseInt(document.getElementById("degree-select").value);
        model = new PolynomialRegression();
        model.fit(xTrainSplit, yTrainSplit, degree);

        console.log("xToPredict tain:", xToPredict);

        let yPredict = model.predict(xToPredict);
        yPredict = yPredict.map(value => Number(value.toFixed(2)));
        //r2 = model.getError();
        //console.log("r2:", r2);
        displayResults(xToPredict, yTrain, yPredict);
    } else if (modelType === "decision-tree") {
        var chart = document.getElementById("tree");
        var canvas = document.getElementById("chart_divRS");
        var { dotStr, predictNode } = testWithChart();
        if (predictNode != null) {
            //var arrHeader = headers.split(",")
            console.log("Predicción: ", headers[headers.length - 1] + ": " + predictNode.value);
            document.getElementById("logRS").innerHTML = `Predicción: ${headers[headers.length - 1]}: ${predictNode.value}`;
        } else {
            document.getElementById("logRS").innerHTML = ``;
        }
        canvas.style.display = "none";  // Cambia display de `bloc` a `none`
        // Asegúrate de que el div `tree` esté visible
        chart.style.display = "block";  // Cambia display de `none` a `block`
        chart.style.width = "100%";
        chart.style.height = "500px";
        chart.style.border = "2px solid rgb(96, 160, 255)";
        chart.style.borderRadius = "10px";

        var parsDot = vis.network.convertDot(dotStr);
        var data = {
            nodes: parsDot.nodes,
            edges: parsDot.edges
        }
        var options = {
            layout: {
                hierarchical: {
                    levelSeparation: 100,
                    nodeSpacing: 100,
                    parentCentralization: true,
                    direction: 'UD', // UD, DU, LR, RL
                    sortMethod: 'directed', // hubsize, directed
                    //shakeTowards: 'roots' // roots, leaves                        
                },
            },
        };
        var network = new vis.Network(chart, data, options);
    } else {
        console.warn("Seleccione un modelo válido para entrenar.");
    }
}

function testWithChart() {
    let dtSt = xTrain;
    let dTree = new DecisionTreeID3(dtSt);
    let root = dTree.train(dTree.dataset);
    var pred = document.getElementById('data').value;
    var arrPred = pred.split(",")
    var predHeader = []
    for (var i = 0; i < headers.length - 1; i++) {
        predHeader.push(headers[i])
    }
    let predict = pred != "" ? dTree.predict([predHeader, arrPred], root) : null;
    return {
        dotStr: dTree.generateDotString(root),
        predictNode: predict
    };
}

// Mostrar resultados y graficar
function displayResults(x, yTrain, yPredict) {
    document.getElementById("logRS").innerHTML = `
      <strong>Datos de Entrenamiento:</strong><br>
      X Train: ${x}<br>
      Y Train: ${yTrain}<br>
      Y Predict: ${yPredict}
    `;
    drawChart(x, yTrain, yPredict);
}

// Función para realizar predicción en un nuevo rango
function makePrediction() {

    const modelType = document.getElementById("model-select").value;
    if (modelType === "linear-regression" || modelType === "polynomial-regression") {

        if (!model || !model.isFit) {
            alert("Primero entrena el modelo.");
            return;
        }
        const rangeInput = document.getElementById("prediction-range").value;
        const newRange = rangeInput.split(',').map(parseFloat);
        const yPredictions = model.predict(newRange);

        document.getElementById("logRS").innerHTML += `<br><strong>Nueva Predicción:</strong> X: ${newRange}, Y: ${yPredictions}`;

    } else if (modelType === "decision-tree") {
        var chart = document.getElementById("tree");
        var canvas = document.getElementById("chart_divRS");
        var { dotStr, predictNode } = testWithChart();
        if (predictNode != null) {
            //var arrHeader = headers.split(",")
            console.log("Predicción: ", headers[headers.length - 1] + ": " + predictNode.value);
            document.getElementById("logRS").innerHTML = `Predicción: ${headers[headers.length - 1]}: ${predictNode.value}`;
        } else {
            document.getElementById("logRS").innerHTML = ``;
        }
        canvas.style.display = "none";  // Cambia display de `bloc` a `none`
        // Asegúrate de que el div `tree` esté visible
        chart.style.display = "block";  // Cambia display de `none` a `block`
        chart.style.width = "100%";
        chart.style.height = "500px";
        chart.style.border = "2px solid rgb(96, 160, 255)";
        chart.style.borderRadius = "10px";

        var parsDot = vis.network.convertDot(dotStr);
        var data = {
            nodes: parsDot.nodes,
            edges: parsDot.edges
        }
        var options = {
            layout: {
                hierarchical: {
                    levelSeparation: 100,
                    nodeSpacing: 100,
                    parentCentralization: true,
                    direction: 'UD', // UD, DU, LR, RL
                    sortMethod: 'directed', // hubsize, directed
                    //shakeTowards: 'roots' // roots, leaves                        
                },
            },
        };
        var network = new vis.Network(chart, data, options);
    } else if (modelType == "neuronal-network") {
        //arrNR
        let arreglodesignado = [2, 4, 3, 2];
        let redNeural = new NeuralNetwork(arreglodesignado);
        for (let i = 0; i < 10000; i++) {
            let numero1 = Math.random();
            let numero2 = Math.random();
            redNeural.Entrenar([numero1, numero2], (numero1 != numero2 ? [1, 0] : [0, 1]));
        }
        let valor1 = arrNR[0];
        let valor2 = arrNR[1];
        console.log("Se ejecuto el codigo para los valores: " + valor1 + "," + valor2);
        document.getElementById("logRS").innerHTML = String("&nbsp;&nbsp;&nbsp;&nbsp;" + redNeural.Predecir([valor1, valor2]));
    }
}

// Gráfico usando Chart.js
function drawChart(xTrain, yTrain, yPredict) {
    const ctx = document.getElementById('chart_divRS').getContext('2d');

    if (window.myChart) {
        window.myChart.destroy();
    }

    const data = {
        labels: xTrain,
        datasets: [
            {
                label: 'Datos de Entrenamiento',
                data: yTrain,
                borderColor: 'blue',
                backgroundColor: 'blue',
                type: 'scatter',
                pointRadius: 4,
            },
            {
                label: 'Predicción',
                data: yPredict,
                borderColor: 'red',
                fill: false,
                type: 'line',
            }
        ]
    };

    const options = {
        responsive: true,
        scales: {
            x: { title: { display: true, text: 'X' } },
            y: { title: { display: true, text: 'Y' } }
        }
    };

    window.myChart = new Chart(ctx, {
        type: 'scatter',
        data: data,
        options: options
    });
}

// Gráfico K-means usando Chart.js
function drawChartKmeans(clusterized_data, clusters) {
    const ctx = document.getElementById('chart_divRS').getContext('2d');

    if (window.myChart) {
        window.myChart.destroy();
    }

    // Datos para los puntos de cada cluster y sus centroides
    const datasets = [];

    // Agregar cada cluster con puntos de color diferente
    clusters.forEach((cluster, index) => {
        const clusterPoints = clusterized_data
            .filter(point => point[1] === cluster[0])
            .map(point => ({ x: point[0], y: 0 })); // Ajuste para graficar en eje X únicamente

        datasets.push({
            label: `Cluster ${index + 1}`,
            data: clusterPoints,
            backgroundColor: cluster[1], // Color asignado al cluster
            pointRadius: 5,
            type: 'scatter'
        });

        // Agregar los centroides de cada cluster
        datasets.push({
            label: `Centroide ${index + 1}`,
            data: [{ x: cluster[0], y: 0 }],
            backgroundColor: '#ff0000', // Color para centroides
            pointStyle: 'rect',
            pointRadius: 6,
            type: 'scatter'
        });
    });

    // Configuración del gráfico
    const options = {
        responsive: true,
        plugins: {
            legend: { display: false }
        },
        scales: {
            x: { title: { display: true, text: 'X' } },
            y: { title: { display: true, text: 'Y' }, min: -1, max: 1 }
        }
    };

    // Crear y dibujar el gráfico
    window.myKmeansChart = new Chart(ctx, {
        type: 'scatter',
        data: { datasets },
        options: options
    });

}

function makePatterns() {
    //clusters;
    //trainingData;
    //iterations;

    if (trainingData.length < clusters) {
        alert(`El numero de clusters (${clusters}) no puede ser menor a la cantidad de datos (${trainingData.length})`)
    }

    var kmeans = new LinearKMeans(clusters, trainingData)
    let clusterized_data = kmeans.clusterize(clusters, trainingData, iterations);

    let clusters_x = new Set([...clusterized_data.map(a => a[1])]);

    clusters_x = Array.from(clusters_x);

    clusters_x.forEach((cluster, i) => {
        clusters_x[i] = [cluster, "#000000".replace(/0/g, function () { return (~~(Math.random() * 16)).toString(16); })]
    });

    drawChartKmeans(clusterized_data, clusters_x);
}

class NodeTree {
    constructor(_value = null, _tag = "", _childs = []) {
        this.id = Math.random().toString(15).substr(3, 12);
        this.tag = _tag;
        this.value = _value;
        this.childs = _childs;
    }
}

class Feature {
    constructor(_attribute, _primaryPosibility, _secondPosibility) {
        this.attribute = _attribute;
        this.entropy = -1;
        this.gain = -1;
        this.primaryCount = 0;
        this.secondaryCount = 0;
        this.primaryPosibility = _primaryPosibility;
        this.secondPosibility = _secondPosibility;
    }

    updateFeature(_posibility) {
        if (_posibility === this.primaryPosibility) {
            this.primaryCount += 1;
        } else if (_posibility === this.secondPosibility) {
            this.secondaryCount += 1;
        } else {
            //error
            return false;
        }
        this.entropy = this.calculateEntropy(this.primaryCount, this.secondaryCount);
        return true;
    }

    calculateEntropy(_p, _n) {
        let entropy = -1;
        if (_p == 0 || _n == 0) {
            return 0;
        }
        entropy = -(_p / (_p + _n)) * Math.log2(_p / (_p + _n))
        entropy += -(_n / (_p + _n)) * Math.log2(_n / (_p + _n))
        return entropy;
    }
}

class Attribute {
    constructor(_attribute) {
        this.attribute = _attribute;
        this.features = [];
        this.infoEntropy = -1;
        this.gain = -1;
        this.index = -1;
    }
}
/**
 * Generate a object to this class and call function train
 */
class DecisionTreeID3 {
    constructor(_dataSet = []) {
        this.dataset = _dataSet
        this.generalEntropy = -1;
        this.primaryCount = -1;
        this.secondaryCount = -1;
        this.primaryPosibility = "";
        this.secondPosibility = "";
        this.root = null;
    }

    /**
     * 
     * @param {Number} _p 
     * @param {Number} _n 
     * @returns {Number}
     */
    calculateEntropy(_p, _n) {
        let entropy = -1;
        if (_p == 0 || _n == 0) {
            return 0;
        }
        entropy = -(_p / (_p + _n)) * Math.log2(_p / (_p + _n))
        entropy += -(_n / (_p + _n)) * Math.log2(_n / (_p + _n))
        return entropy;
    }

    /**
     * This funcition only acept:
     *      - The data set is a matrix and contain the header
     *      - The result column must be in the matrix last column
     * this funcition return the root of the tree generated
     * @param {any[]} _dataset 
     * @param {Integer} _start 
     * @returns {NodeTree}
     */
    train(_dataset, _start = 0) {
        // We going to train the algorithm
        // First we going to calculate entropy of data set
        let columnResult = _dataset[0].length - 1;
        this.calculateGeneralEntropy(_dataset, columnResult);

        /**
         * Second we going to classifier every feature and calculate the entropy of every feature inside of data set
         * This process is realized for every Attribute
         * */
        let numberAttributes = _dataset[0].length;
        let gainAttribute = []
        for (let i = _start; i < numberAttributes; i++) {
            if (i === columnResult) continue;
            let attribute = new Attribute(_dataset[0][i]);
            attribute.index = i;
            attribute.features = this.classifierFeatures(_dataset, i, columnResult);
            attribute.infoEntropy = this.calculateInformationEntropy(attribute.features);
            attribute.gain = this.calculateGain(this.generalEntropy, attribute.infoEntropy);
            gainAttribute.push(attribute);
        }
        if (gainAttribute.length == 0) {
            return null;
        }
        /**
         * Third we going to select the best attribute
         */
        let selectedGain = this.selectBestFeature(gainAttribute);

        /**
         * We going to create a node with the best attribute selected
         */

        let parentNode = new NodeTree(gainAttribute[selectedGain].attribute);
        gainAttribute[selectedGain].features.map(feat => {
            let childNode = new NodeTree(null);
            if (feat.entropy == 0) {
                childNode.value = feat.primaryCount == 0 ? feat.secondPosibility : feat.primaryPosibility;
            } else {
                let newDataSet = _dataset.filter((split, index) => (split[gainAttribute[selectedGain].index] === feat.attribute) || index == 0)
                if (_start < 4 && newDataSet.length > 2) childNode = this.train(newDataSet, _start + 1);
            }
            childNode.tag = feat.attribute;
            parentNode.childs.push(childNode);
        });
        return parentNode;

    }

    /**
     * Simple function to predict a data
     * @param {any[]} _predict 
     * @param {NodeTree} _root 
     * @returns 
     */
    predict(_predict, _root) {
        return this.recursivePredict(_predict, _root);
    }

    /**
     * Simple function
     * @param {any[]} _predict 
     * @param {NodeTree} _node 
     * @returns 
     */
    recursivePredict(_predict, _node) {
        if (_node.childs.length == 0) return _node;
        for (let index = 0; index < _predict[0].length; index++) {
            if (_predict[0][index] === _node.value) {
                //if(this.childs.length == 0) return
                for (let i = 0; i < _node.childs.length; i++) {
                    if (_node.childs[i].tag === _predict[1][index]) {
                        return this.recursivePredict(_predict, _node.childs[i])
                    }
                }
            }
        }
        return null;
    }

    /**
     * 
     * @param {String Array} _dataset 
     * @param {Integer} indexResult this attribute is to indicate the result column in the data set
     * @returns {Float}
     */
    calculateGeneralEntropy(_dataset, indexResult) {
        let att1 = {
            tag: "",
            count: 0
        }
        let att2 = {
            tag: "",
            count: 0
        }
        let header = false;
        _dataset.map(f => {
            if (header) {
                if (!att1.tag) {
                    att1.tag = f[indexResult];
                    att1.count += 1;
                } else if (!att2.tag && f[indexResult] != att1.tag) {
                    att2.tag = f[indexResult];
                    att2.count += 1;
                } else if (att1.tag === f[indexResult]) {
                    att1.count += 1;
                } else if (att2.tag === f[indexResult]) {
                    att2.count += 1;
                }
            } else {
                header = true;
            }

        });
        this.primaryPosibility = att1.tag;
        this.secondPosibility = att2.tag;
        this.primaryCount = att1.count;
        this.secondaryCount = att2.count;
        this.generalEntropy = this.calculateEntropy(att1.count, att2.count);
        return this.generalEntropy;
    }

    /**
     * 
     * @param {string[]} _dataset 
     * @param {Integer} indexFeature 
     * @param {Integer} indexResult 
     * @returns 
     */
    classifierFeatures(_dataset, indexFeature, indexResult) {
        let features = []
        let header = false;
        _dataset.map(f => {
            if (header) {
                let index = features.findIndex(t => t.attribute === f[indexFeature]);
                if (index > -1) {
                    features[index].updateFeature(f[indexResult]);
                } else {
                    let feat = new Feature(f[indexFeature], this.primaryPosibility, this.secondPosibility);
                    feat.updateFeature(f[indexResult]);
                    features.push(feat);
                }
            } else {
                header = true;
            }
        });
        return features;
    }

    /**
     * 
     * @param {Feature[]} _features 
     * @returns {Number}
     */
    calculateInformationEntropy(_features) {
        let infoEntropy = 0;
        _features.map(f => {
            infoEntropy += ((f.primaryCount + f.secondaryCount) / (this.primaryCount + this.secondaryCount)) * f.entropy;
        })
        return infoEntropy;
    }

    /**
     * 
     * @param {Number} _generalEntropy 
     * @param {Number} _infoEntropy 
     * @returns {Number}
     */
    calculateGain(_generalEntropy, _infoEntropy) {
        let gain = _generalEntropy - _infoEntropy;
        return gain;
    }

    /**
     * Select the best Attribute with the max gain factor and return the index of the feature selected.
     * @param {Attribute[]} _attributes 
     * @returns {Integer}
     */
    selectBestFeature(_attributes) {
        let index = -1;
        let best = -1000;
        _attributes.map((feature, indexFeature) => {
            if (feature.gain > best) {
                best = feature.gain;
                index = indexFeature;
            }

        })
        return index
    }

    /**
     * this function is for create a string with the dot structure to create the graphic tree.
     * @param {NodeTree} _root 
     * @returns {string}
     */
    generateDotString(_root) {
        let dotStr = "{";
        dotStr += this.recursiveDotString(_root);
        dotStr += "}";
        return dotStr;
    }

    /**
     * this function is for travel the tree structure.
     * @param {NodeTree} _root 
     * @param {string} _idParent 
     * @returns {string}
     */
    recursiveDotString(_root, _idParent = "") {
        let dotStr = "";
        if (!_root) return "";
        dotStr += `${_root.id} [label="${_root.value}"];`;
        dotStr += _idParent ? `${_idParent}--${_root.id}` : "";
        dotStr += _root.tag ? `[label="${_root.tag}"];` : "";
        _root.childs.map(child => {
            dotStr += this.recursiveDotString(child, _root.id);
        });
        return dotStr;
    }
}

//------- Red neuronal -------------
class Matriz {

    constructor(rows, cols) {
        this.rows = rows;
        this.cols = cols;

        this.data = [];
        for (let i = 0; i < this.rows; i++) {
            this.data[i] = [];
            for (let j = 0; j < this.cols; j++)
                this.data[i][j] = 0;
        }


    }

    // http://github.com/AlexDenver

    static multiplicar(m1, m2) {

        if (m1.cols !== m2.rows) {
            console.log("Cannot Operate, Check Matriz Multiplication Rules.");
            return undefined;
        } else {
            let result = new Matriz(m1.rows, m2.cols);

            for (let i = 0; i < result.rows; i++)
                for (let j = 0; j < result.cols; j++) {
                    let sum = 0;
                    for (let k = 0; k < m1.cols; k++) {
                        sum += m1.data[i][k] * m2.data[k][j];
                    }
                    result.data[i][j] = sum;
                }
            return result;

        }
    }
    multiplicar(n) {
        if (n instanceof Matriz) {

            for (let i = 0; i < this.rows; i++)
                for (let j = 0; j < this.cols; j++)
                    this.data[i][j] *= n.data[i][j];

        } else {
            for (let i = 0; i < this.rows; i++)
                for (let j = 0; j < this.cols; j++)
                    this.data[i][j] *= n;
        }
    }

    summar(n) {
        if (n instanceof Matriz) {
            for (let i = 0; i < this.rows; i++)
                for (let j = 0; j < this.cols; j++)
                    this.data[i][j] += n.data[i][j];
        } else {
            for (let i = 0; i < this.rows; i++)
                for (let j = 0; j < this.cols; j++)
                    this.data[i][j] += n;
        }

    }

    static resstar(a, b) {
        let res = new Matriz(a.rows, a.cols);
        for (let i = 0; i < a.rows; i++)
            for (let j = 0; j < a.cols; j++)
                res.data[i][j] = a.data[i][j] - b.data[i][j];
        return res;
    }

    mapear(func) {
        for (let i = 0; i < this.rows; i++)
            for (let j = 0; j < this.cols; j++) {
                let val = this.data[i][j];
                this.data[i][j] = func(val);
            }
    }

    static mapear(m, func) {
        for (let i = 0; i < m.rows; i++)
            for (let j = 0; j < m.cols; j++) {
                let val = m.data[i][j];
                m.data[i][j] = func(val);
            }
        return m;
    }

    tirar_random() {
        for (let i = 0; i < this.rows; i++)
            for (let j = 0; j < this.cols; j++)
                this.data[i][j] = (Math.random() * 2) - 1;  //between -1 and 1
    }

    static transpuesta(m) {
        let res = new Matriz(m.cols, m.rows);
        for (let i = 0; i < m.rows; i++)
            for (let j = 0; j < m.cols; j++)
                res.data[j][i] = m.data[i][j];
        return res;
    }

    imprimir() {
        console.table(this.data);
    }

    convert_to_array() {
        let arr = [];
        for (let i = 0; i < this.rows; i++)
            for (let j = 0; j < this.cols; j++)
                arr.push(this.data[i][j]);
        return arr;
    }

    static get_array(array) {
        let m = new Matriz(array.length, 1);
        for (let i = 0; i < array.length; i++) {
            m.data[i][0] = array[i];
        }
        // m.print();
        return m;
    }


}


class LayerLink {

    constructor(prevNode_count, node_count) {
        this.weights = new Matriz(node_count, prevNode_count);
        this.bias = new Matriz(node_count, 1);
        this.weights.tirar_random();
        this.bias.tirar_random();

        //http://github.com/AlexDenver

        //console.table(this.weights.data)
        //console.table(this.bias.data)
    }

    actualizar_Weights(weights) {
        this.weights = weights;
    }
    obtener_Weights() {
        return this.weights;
    }
    obtener_Bias() {
        return this.bias;
    }
    summar(deltaWeight, bias) {
        this.weights.summar(deltaWeight);
        this.bias.summar(bias);
    }

}


class NeuralNetwork {

    //http://github.com/AlexDenver

    constructor(layers, options) {
        if (layers.length < 2) {
            console.error("Neural Network Needs Atleast 2 Layers To Work.");
            return { layers: layers };
        }
        this.options = {
            activation: function (x) {
                return (1 / (1 + Math.exp(-x)));
            },
            derivative: function (y) {
                return (y * (1 - y));
            }
        }
        this.learning_rate = 0.1;
        if (options) {
            if (options.learning_rate)
                this.Set_aprendizaje_rate(parseFloat(options.learning_rate));
            if (options.activation && options.derivative &&
                options.activation instanceof Function &&
                options.derivative instanceof Function) {
                this.options.activation = options.activation;
                this.options.derivative = options.derivative;
            } else {
                console.error("Check Documentation to Learn How To Set Custom Activation Function");
                console.warn("Documentation Link: http://github.com/AlexDenver")
                return { options: options };
            }
        }
        this.layerCount = layers.length - 1;   // Ignoring Output Layer.
        this.inputs = layers[0];
        this.output_nodes = layers[layers.length - 1];
        this.layerLink = [];
        for (let i = 1, j = 0; j < (this.layerCount); i++, j++) {
            if (layers[i] <= 0) {
                console.error("A Layer Needs To Have Atleast One Node (Neuron).");
                return { layers: layers };
            }
            this.layerLink[j] = new LayerLink(layers[j], layers[i]);    // Previous Layer Nodes & Current Layer Nodes
        }

    }

    Predecir(input_array) {

        if (input_array.length !== this.inputs) {
            console.error(`This Instance of NeuralNetwork Expects ${this.inputs} Inputs, ${input_array.length} Provided.`);
            return { inputs: input_array };
        }
        let input = Matriz.get_array(input_array);
        let layerResult = input;
        for (let i = 0; i < this.layerCount; i++) {
            layerResult = Matriz.multiplicar(this.layerLink[i].obtener_Weights(), layerResult);
            layerResult.summar(this.layerLink[i].obtener_Bias());
            layerResult.mapear(this.options.activation);
        }
        // The Last Layer Result will be the Final Output.
        return layerResult.convert_to_array();
    }

    Set_aprendizaje_rate(n) {
        if (n > 1 && n < 100) {
            n = n / 100;
        } else {
            console.error("Set Learning Rate Between (0 - 1) or (1 - 100)");
            return;
        }
        if (n > 0.3) {
            console.warn("It is recommended to Set Lower Learning Rates");
        }
        this.learning_rate = n;
    }

    Entrenar(input_array, target_array) {
        if (input_array.length !== this.inputs) {
            console.error(`This Instance of NeuralNetwork Expects ${this.inputs} Inputs, ${input_array.length} Provided.`);
            return { inputs: input_array };
        }
        if (target_array.length !== this.output_nodes) {
            console.error(`This Instance of NeuralNetwork Expects ${this.output_nodes} Outputs, ${target_array.length} Provided.`);
            return { outputs: target_array };
        }
        let input = Matriz.get_array(input_array);
        // Array to Store/Track each Layer Weighted Result (sum)
        let layerResult = [];
        layerResult[0] = input;  // Since input is First Layer.
        // Predicting the Result for Given Input, Store Output of each Consequent layer
        for (let i = 0; i < this.layerCount; i++) {
            layerResult[i + 1] = Matriz.multiplicar(this.layerLink[i].obtener_Weights(), layerResult[i]);
            layerResult[i + 1].summar(this.layerLink[i].obtener_Bias());
            layerResult[i + 1].mapear(this.options.activation);
        }

        let targets = Matriz.get_array(target_array);
        // Variables to Store Errors and Gradients at each Layer.
        let layerErrors = [];
        let gradients = [];

        // Calculate Actual Error based on Target.
        layerErrors[this.layerCount] = Matriz.resstar(targets, layerResult[this.layerCount]);

        // Correcting and Recalculating Error for each Layer
        for (let i = this.layerCount; i > 0; i--) {
            // Calculate the Layer Gradient 
            // dyE/dyW = learning_rate * layerError * sigmoid(x) * (1-sigmoid(x)); 
            // NOTE: dsigmoid = sigmoid(x) * (1-sigmoid(x) ie derivative of sigmoid

            gradients[i] = Matriz.mapear(layerResult[i], this.options.derivative);
            gradients[i].multiplicar(layerErrors[i]);
            gradients[i].multiplicar(this.learning_rate);

            // Calculate the Changes to be made to the weighs
            let hidden_T = Matriz.transpuesta(layerResult[i - 1]);
            let weight_ho_deltas = Matriz.multiplicar(gradients[i], hidden_T);

            // Update the Weights and Gradient According to Deltas & Gradient.
            this.layerLink[i - 1].summar(weight_ho_deltas, gradients[i]);

            // Calculate the Previous Layer Errors (Proportional Error based on Current Layer Error.)
            // NOTE: We are Backpropogating, Therefore we are going backwards 1 step (i.e. i-1)
            layerErrors[i - 1] = Matriz.multiplicar(Matriz.transpuesta(this.layerLink[i - 1].obtener_Weights()), layerErrors[i]);
        }

    }

}
//----------------------------------

//----------------k-means-linear------------------
class LinearKMeans {
    constructor() {
        this.k = 3
        this.data = []
    }

    clusterize(k, data, iterations) {
        //console.log(iterations)
        let clusters_proposals = []
        this.data = data
        //console.log(k, data)

        for (let i = 0; i < iterations; i++) {
            let clusters = []


            //pendiente agregar iterations
            //let copy_data = [...this.data]
            for (let i = 0; i < k; i++) {
                let c = data[Math.floor(Math.random() * data.length)]
                while (clusters.findIndex(x => x === c) != -1) {
                    c = data[Math.floor(Math.random() * data.length)]
                }
                clusters.push(c)
            }

            clusters = clusters.sort(function (a, b) { return (a > b) ? 1 : ((b > a) ? -1 : 0); })
            //console.log('Clusters: ', clusters)

            //distancias [punto,cluster,distancia]
            let distances = []
            let clustered_data = []
            let closer_cluster = 0, closer_distance = 0, first_distance = true;
            data.forEach(point => {
                closer_cluster = 0, closer_distance = 0, first_distance = true;

                clusters.forEach(c => {
                    let distance = this.distance(point, c)
                    if (first_distance) {
                        closer_distance = Math.abs(distance)
                        closer_cluster = c
                        first_distance = !first_distance
                    } else {
                        if (Math.abs(distance) < closer_distance) {
                            closer_distance = Math.abs(distance)
                            closer_cluster = c
                        }

                    }
                    distances.push([point, c, distance])


                })
                clustered_data.push([point, closer_cluster, closer_distance])
            });


            //calcular medias y varianza

            let previous_cluster_stats = []
            let total_variance = 0
            let cluster_stats = []
            do {
                previous_cluster_stats = cluster_stats
                cluster_stats = []
                total_variance = 0
                // [mean,variance]
                clusters.forEach((c, i) => {
                    let data = clustered_data.filter((cd) => cd[1] == c)
                    let data_points = data.map((dp) => dp[0])
                    cluster_stats.push(this.calculateMeanVariance(data_points))
                    total_variance += data_points[1]
                    clusters[i] = cluster_stats[i][0]
                });

                if (Number.isNaN(total_variance)) {
                    total_variance = 0
                }


                // Recalcular distancias y agrupaciones

                distances = []
                clustered_data = []
                closer_cluster = 0, closer_distance = 0, first_distance = true;
                data.forEach(point => {
                    closer_cluster = 0, closer_distance = 0, first_distance = true;

                    clusters.forEach(c => {
                        let distance = this.distance(point, c)
                        if (first_distance) {
                            closer_distance = Math.abs(distance)
                            closer_cluster = c
                            first_distance = !first_distance
                        } else {
                            if (Math.abs(distance) < closer_distance) {
                                closer_distance = Math.abs(distance)
                                closer_cluster = c
                            }

                        }
                        distances.push([point, c, distance])

                    })
                    clustered_data.push([point, closer_cluster, closer_distance])


                });

                //console.log(clusters)
                //console.table(clustered_data)
            } while (JSON.stringify(previous_cluster_stats) != JSON.stringify(cluster_stats))
            clusters_proposals.push([total_variance, clusters, clustered_data])
        }

        // Posibles clusters
        // console.log(clusters_proposals)

        clusters_proposals.sort(function (a, b) { return (a[0] > b[0]) ? 1 : ((b[0] > a[0]) ? -1 : 0); })
        //console.table(distances)


        return clusters_proposals[0][2]

    }

    distance(point_a, point_b) {
        return point_b - point_a
    }

    calculateMeanVariance(arr) {

        function getVariance(arr, mean) {
            return arr.reduce(function (pre, cur) {
                pre = pre + Math.pow((cur - mean), 2);
                return pre;
            }, 0)
        }

        var meanTot = arr.reduce(function (pre, cur) {
            return pre + cur;
        })

        var total = getVariance(arr, meanTot / arr.length);

        var res = {
            mean: meanTot / arr.length,
            variance: total / arr.length
        }


        return [res.mean, res.variance]
    }
}

//esto es un comentario extra