// main.js
document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("file-input").addEventListener("change", handleFileUpload);
    document.getElementById("model-select").addEventListener("change", resetModel);

    // Otros eventos e inicializaciones
});

let model;

function resetModel() {
    document.getElementById("logRS").innerHTML = "";
    document.getElementById("chart_divRS").style.display = "none";
    document.getElementById("tree").style.display = "none";
    if (window.myChart) window.myChart.destroy();
}

// Manejo de la carga del archivo CSV
function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            const csvData = e.target.result;
            loadModelData(csvData);
        };
        reader.readAsText(file);
    }
}

// Seleccionar y cargar datos de acuerdo con el modelo seleccionado
function loadModelData(data) {
    const modelType = document.getElementById("model-select").value;
    let parsedData;

    if (modelType === "linear-regression") {
        parsedData = parseCSVDataLinearRegression(data);
        model = new LinearRegression();
    } else if (modelType === "polynomial-regression") {
        parsedData = parseCSVDataPolynomialRegression(data);
        model = new PolynomialRegression();
    } else if (modelType === "decision-tree") {
        parsedData = parseCSVDataDecisionTree(data);
        model = new DecisionTreeID3(parsedData.trainData);
    }

    trainModel(parsedData);
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


function trainModel(parsedData) {
    const modelType = document.getElementById("model-select").value;
    const trainPercent = parseInt(document.getElementById("train-test-split").value);
    console.log("trainPercent: ", trainPercent);

    if (modelType === "linear-regression") {
        model.fit(parsedData.xTrain, parsedData.yTrain);
        const yPredict = model.predict(parsedData.xTrain);
        displayResults(parsedData.xTrain, parsedData.yTrain, yPredict);
    } else if (modelType === "polynomial-regression") {
        const degree = parseInt(document.getElementById("degree-select").value);
        model.fit(parsedData.xTrain, parsedData.yTrain, degree);
        const yPredict = model.predict(parsedData.xToPredict);
        displayResults(parsedData.xToPredict, parsedData.yTrain, yPredict);
    } else if (modelType === "decision-tree") {
        model.train(parsedData.trainData);
        displayTree(model.root);
        const prediction = model.predict(parsedData.xToPredict);
        document.getElementById("logRS").innerHTML += `<br><strong>Predicción:</strong> ${prediction}`;
    }
}

function displayResults(x, yTrain, yPredict) {
    document.getElementById("logRS").innerHTML = `
        <strong>Datos de Entrenamiento:</strong><br>
        X Train: ${x}<br>
        Y Train: ${yTrain}<br>
        Y Predict: ${yPredict}
    `;
    drawChart(x, yTrain, yPredict);
}

function displayTree(rootNode) {
    document.getElementById("tree").style.display = "block";
    const dotStr = model.generateDotString(rootNode);
    document.getElementById("logRS").innerHTML = `
        <strong>Árbol de Decisión:</strong><br>
        DOT String: ${dotStr}
    `;
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

// Función para realizar predicción en un nuevo rango
function makePrediction() {
    if (!model || !model.isFit) {
        alert("Primero entrena el modelo.");
        return;
    }

    const rangeInput = document.getElementById("prediction-range").value;
    const newRange = rangeInput.split(',').map(parseFloat);
    const yPredictions = model.predict(newRange);

    document.getElementById("logRS").innerHTML += `<br><strong>Nueva Predicción:</strong> X: ${newRange}, Y: ${yPredictions}`;
}