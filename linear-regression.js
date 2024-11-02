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

// Función para procesar CSV específico para Regresión Lineal
function parseCSVDataLinearRegression(data) {
    const lines = data.trim().split('\n').slice(1);
    let xTrain = [];
    let yTrain = [];

    lines.forEach(line => {
        const [x, y] = line.split(';').map(parseFloat);
        xTrain.push(x);
        yTrain.push(y);
    });

    return { xTrain, yTrain };
}