// Medical Imaging Cross-Validation Interactive Animation
// Demonstrates proper patient-level data splitting

class TrustCVAnimation {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.patients = [];
        this.animationStep = 0;
        this.isAnimating = false;
        this.currentDemo = 'splitting';
        
        // Set canvas size
        this.canvas.width = 800;
        this.canvas.height = 500;
        
        // Colors matching KI theme
        this.colors = {
            primary: '#8B1538',    // KI Plum
            secondary: '#E9423E',  // KI Red
            success: '#4CAF50',
            error: '#F44336',
            warning: '#FF9800',
            train: '#3498DB',
            test: '#E74C3C',
            neutral: '#95A5A6',
            background: '#F5F5F5',
            text: '#2C3E50'
        };
        
        this.initializePatients();
    }
    
    initializePatients() {
        // Create sample patients with multiple images each
        const patientData = [
            { id: 'P001', name: 'Patient A', images: 3, condition: 'healthy' },
            { id: 'P002', name: 'Patient B', images: 4, condition: 'disease' },
            { id: 'P003', name: 'Patient C', images: 2, condition: 'healthy' },
            { id: 'P004', name: 'Patient D', images: 3, condition: 'disease' },
            { id: 'P005', name: 'Patient E', images: 4, condition: 'healthy' },
            { id: 'P006', name: 'Patient F', images: 2, condition: 'disease' },
            { id: 'P007', name: 'Patient G', images: 3, condition: 'healthy' },
            { id: 'P008', name: 'Patient H', images: 3, condition: 'disease' }
        ];
        
        let x = 50;
        let y = 100;
        
        this.patients = patientData.map((patient, index) => {
            if (index === 4) {
                x = 50;
                y = 250;
            }
            const patientObj = {
                ...patient,
                x: x,
                y: y,
                width: 150,
                height: 120,
                images: [],
                inTraining: null,
                highlighted: false
            };
            
            // Create image representations
            for (let i = 0; i < patient.images; i++) {
                patientObj.images.push({
                    id: `${patient.id}_img${i + 1}`,
                    x: x + 10 + (i * 35),
                    y: y + 40,
                    width: 30,
                    height: 30
                });
            }
            
            x += 180;
            return patientObj;
        });
    }
    
    clear() {
        this.ctx.fillStyle = this.colors.background;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }
    
    drawTitle(title) {
        this.ctx.fillStyle = this.colors.text;
        this.ctx.font = 'bold 24px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(title, this.canvas.width / 2, 40);
    }
    
    drawPatient(patient) {
        // Draw patient container
        this.ctx.strokeStyle = patient.highlighted ? this.colors.warning : this.colors.neutral;
        this.ctx.lineWidth = patient.highlighted ? 3 : 1;
        
        if (patient.inTraining === true) {
            this.ctx.fillStyle = this.colors.train + '30';
        } else if (patient.inTraining === false) {
            this.ctx.fillStyle = this.colors.test + '30';
        } else {
            this.ctx.fillStyle = '#FFFFFF';
        }
        
        this.ctx.fillRect(patient.x, patient.y, patient.width, patient.height);
        this.ctx.strokeRect(patient.x, patient.y, patient.width, patient.height);
        
        // Draw patient label
        this.ctx.fillStyle = this.colors.text;
        this.ctx.font = 'bold 14px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto';
        this.ctx.textAlign = 'left';
        this.ctx.fillText(patient.name, patient.x + 10, patient.y + 20);
        
        // Draw condition label
        this.ctx.font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto';
        this.ctx.fillStyle = patient.condition === 'disease' ? this.colors.error : this.colors.success;
        this.ctx.fillText(patient.condition.toUpperCase(), patient.x + 10, patient.y + 35);
        
        // Draw images
        patient.images.forEach((img, index) => {
            // Image container
            this.ctx.fillStyle = patient.condition === 'disease' ? this.colors.error + '40' : this.colors.success + '40';
            this.ctx.fillRect(img.x, img.y, img.width, img.height);
            
            // Image border
            this.ctx.strokeStyle = this.colors.neutral;
            this.ctx.lineWidth = 1;
            this.ctx.strokeRect(img.x, img.y, img.width, img.height);
            
            // Image icon (simplified MRI/CT representation)
            this.ctx.fillStyle = this.colors.text;
            this.ctx.font = '10px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto';
            this.ctx.textAlign = 'center';
            this.ctx.fillText('IMG', img.x + img.width/2, img.y + img.height/2 + 3);
        });
    }
    
    drawSplitLine() {
        this.ctx.strokeStyle = this.colors.primary;
        this.ctx.lineWidth = 3;
        this.ctx.setLineDash([10, 5]);
        this.ctx.beginPath();
        this.ctx.moveTo(this.canvas.width / 2, 80);
        this.ctx.lineTo(this.canvas.width / 2, 420);
        this.ctx.stroke();
        this.ctx.setLineDash([]);
        
        // Labels
        this.ctx.fillStyle = this.colors.train;
        this.ctx.font = 'bold 18px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('TRAINING SET', this.canvas.width / 4, 70);
        
        this.ctx.fillStyle = this.colors.test;
        this.ctx.fillText('TEST SET', 3 * this.canvas.width / 4, 70);
    }
    
    drawMessage(message, type = 'info') {
        let bgColor = this.colors.primary + '20';
        let borderColor = this.colors.primary;
        
        if (type === 'error') {
            bgColor = this.colors.error + '20';
            borderColor = this.colors.error;
        } else if (type === 'success') {
            bgColor = this.colors.success + '20';
            borderColor = this.colors.success;
        }
        
        // Message box
        this.ctx.fillStyle = bgColor;
        this.ctx.fillRect(50, 440, this.canvas.width - 100, 40);
        this.ctx.strokeStyle = borderColor;
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(50, 440, this.canvas.width - 100, 40);
        
        // Message text
        this.ctx.fillStyle = this.colors.text;
        this.ctx.font = '14px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(message, this.canvas.width / 2, 465);
    }
    
    // Animation: Correct Patient-Level Splitting
    animateCorrectSplitting() {
        const steps = [
            () => {
                this.clear();
                this.drawTitle('Correct: Patient-Level Splitting');
                this.patients.forEach(p => {
                    p.inTraining = null;
                    p.highlighted = false;
                    this.drawPatient(p);
                });
                this.drawMessage('Starting patient-level data splitting...', 'info');
            },
            () => {
                // Assign first 5 patients to training
                this.patients.slice(0, 5).forEach(p => p.inTraining = true);
                this.clear();
                this.drawTitle('Correct: Patient-Level Splitting');
                this.patients.forEach(p => this.drawPatient(p));
                this.drawSplitLine();
                this.drawMessage('Assigning complete patients to training set', 'success');
            },
            () => {
                // Assign remaining patients to test
                this.patients.slice(5).forEach(p => p.inTraining = false);
                this.clear();
                this.drawTitle('Correct: Patient-Level Splitting');
                this.patients.forEach(p => this.drawPatient(p));
                this.drawSplitLine();
                this.drawMessage('Remaining patients go to test set - No data leakage!', 'success');
            },
            () => {
                // Highlight a patient to show all images stay together
                this.patients[1].highlighted = true;
                this.clear();
                this.drawTitle('Correct: Patient-Level Splitting');
                this.patients.forEach(p => this.drawPatient(p));
                this.drawSplitLine();
                this.drawMessage('All images from Patient B stay in training set together', 'success');
            }
        ];
        
        if (this.animationStep < steps.length) {
            steps[this.animationStep]();
            this.animationStep++;
        } else {
            this.animationStep = 0;
            this.isAnimating = false;
        }
    }
    
    // Animation: Incorrect Random Splitting
    animateIncorrectSplitting() {
        const steps = [
            () => {
                this.clear();
                this.drawTitle('Incorrect: Random Image Splitting');
                this.patients.forEach(p => {
                    p.inTraining = null;
                    p.highlighted = false;
                    this.drawPatient(p);
                });
                this.drawMessage('Starting random image splitting (WRONG!)', 'error');
            },
            () => {
                // Randomly assign images (simulated - showing mixed assignment)
                this.patients[0].inTraining = true;  // Some images in training
                this.patients[1].inTraining = null;   // Mixed - WRONG!
                this.patients[2].inTraining = false;  // Some images in test
                this.patients[3].inTraining = true;
                this.patients[4].inTraining = null;   // Mixed - WRONG!
                this.patients[5].inTraining = false;
                this.patients[6].inTraining = true;
                this.patients[7].inTraining = null;   // Mixed - WRONG!
                
                this.clear();
                this.drawTitle('Incorrect: Random Image Splitting');
                this.patients.forEach(p => this.drawPatient(p));
                this.drawSplitLine();
                this.drawMessage('Random splitting breaks patient grouping!', 'error');
            },
            () => {
                // Highlight mixed patients
                this.patients[1].highlighted = true;
                this.patients[4].highlighted = true;
                this.patients[7].highlighted = true;
                
                this.clear();
                this.drawTitle('Incorrect: Random Image Splitting');
                this.patients.forEach(p => this.drawPatient(p));
                this.drawSplitLine();
                this.drawMessage('DATA LEAKAGE: Same patient in both train and test!', 'error');
            },
            () => {
                this.clear();
                this.drawTitle('Incorrect: Random Image Splitting');
                this.patients.forEach(p => this.drawPatient(p));
                this.drawSplitLine();
                
                // Draw warning arrows
                this.ctx.strokeStyle = this.colors.error;
                this.ctx.lineWidth = 2;
                this.ctx.beginPath();
                // Arrow from patient to both sides
                this.ctx.moveTo(this.patients[1].x + 75, this.patients[1].y + 60);
                this.ctx.lineTo(200, this.patients[1].y + 60);
                this.ctx.moveTo(this.patients[1].x + 75, this.patients[1].y + 60);
                this.ctx.lineTo(600, this.patients[1].y + 60);
                this.ctx.stroke();
                
                this.drawMessage('Model will memorize patient patterns - Overfit!', 'error');
            }
        ];
        
        if (this.animationStep < steps.length) {
            steps[this.animationStep]();
            this.animationStep++;
        } else {
            this.animationStep = 0;
            this.isAnimating = false;
        }
    }
    
    // Animation: K-Fold Cross-Validation
    animateKFoldCV() {
        const folds = [
            [0, 1],  // Fold 1 as test
            [2, 3],  // Fold 2 as test
            [4, 5],  // Fold 3 as test
            [6, 7]   // Fold 4 as test
        ];
        
        const steps = [
            () => {
                this.clear();
                this.drawTitle('K-Fold Cross-Validation (K=4)');
                this.patients.forEach(p => {
                    p.inTraining = null;
                    p.highlighted = false;
                    this.drawPatient(p);
                });
                this.drawMessage('4-Fold CV: Each fold serves as test set once', 'info');
            },
            () => {
                // Fold 1
                this.patients.forEach((p, i) => {
                    p.inTraining = !folds[0].includes(i);
                    p.highlighted = folds[0].includes(i);
                });
                this.clear();
                this.drawTitle('K-Fold Cross-Validation - Fold 1');
                this.patients.forEach(p => this.drawPatient(p));
                this.drawMessage('Fold 1: Patients A & B as test, others as training', 'info');
            },
            () => {
                // Fold 2
                this.patients.forEach((p, i) => {
                    p.inTraining = !folds[1].includes(i);
                    p.highlighted = folds[1].includes(i);
                });
                this.clear();
                this.drawTitle('K-Fold Cross-Validation - Fold 2');
                this.patients.forEach(p => this.drawPatient(p));
                this.drawMessage('Fold 2: Patients C & D as test, others as training', 'info');
            },
            () => {
                // Fold 3
                this.patients.forEach((p, i) => {
                    p.inTraining = !folds[2].includes(i);
                    p.highlighted = folds[2].includes(i);
                });
                this.clear();
                this.drawTitle('K-Fold Cross-Validation - Fold 3');
                this.patients.forEach(p => this.drawPatient(p));
                this.drawMessage('Fold 3: Patients E & F as test, others as training', 'info');
            },
            () => {
                // Fold 4
                this.patients.forEach((p, i) => {
                    p.inTraining = !folds[3].includes(i);
                    p.highlighted = folds[3].includes(i);
                });
                this.clear();
                this.drawTitle('K-Fold Cross-Validation - Fold 4');
                this.patients.forEach(p => this.drawPatient(p));
                this.drawMessage('Fold 4: Patients G & H as test, others as training', 'info');
            },
            () => {
                this.clear();
                this.drawTitle('K-Fold Cross-Validation Complete');
                this.patients.forEach(p => {
                    p.inTraining = null;
                    p.highlighted = false;
                    this.drawPatient(p);
                });
                this.drawMessage('All patients tested once, trained K-1 times', 'success');
            }
        ];
        
        if (this.animationStep < steps.length) {
            steps[this.animationStep]();
            this.animationStep++;
        } else {
            this.animationStep = 0;
            this.isAnimating = false;
        }
    }
    
    // Animation: Temporal Validation
    animateTemporalValidation() {
        // Add timestamps to patients
        const timestamps = ['Jan 2023', 'Feb 2023', 'Mar 2023', 'Apr 2023', 
                          'May 2023', 'Jun 2023', 'Jul 2023', 'Aug 2023'];
        
        const steps = [
            () => {
                this.clear();
                this.drawTitle('Temporal Validation for Time-Series Data');
                
                // Draw timeline
                this.ctx.strokeStyle = this.colors.neutral;
                this.ctx.lineWidth = 2;
                this.ctx.beginPath();
                this.ctx.moveTo(50, 250);
                this.ctx.lineTo(750, 250);
                this.ctx.stroke();
                
                // Draw patients on timeline
                this.patients.forEach((p, i) => {
                    const x = 80 + (i * 80);
                    const y = 200;
                    
                    // Patient box
                    this.ctx.fillStyle = '#FFFFFF';
                    this.ctx.fillRect(x - 30, y, 60, 40);
                    this.ctx.strokeStyle = this.colors.neutral;
                    this.ctx.strokeRect(x - 30, y, 60, 40);
                    
                    // Patient label
                    this.ctx.fillStyle = this.colors.text;
                    this.ctx.font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto';
                    this.ctx.textAlign = 'center';
                    this.ctx.fillText(p.name.split(' ')[1], x, y + 25);
                    
                    // Timestamp
                    this.ctx.font = '10px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto';
                    this.ctx.fillText(timestamps[i], x, y + 55);
                    
                    // Timeline marker
                    this.ctx.beginPath();
                    this.ctx.arc(x, 250, 4, 0, 2 * Math.PI);
                    this.ctx.fillStyle = this.colors.primary;
                    this.ctx.fill();
                });
                
                this.drawMessage('Medical data collected over time', 'info');
            },
            () => {
                this.clear();
                this.drawTitle('Temporal Validation - Training Window');
                
                // Draw timeline
                this.ctx.strokeStyle = this.colors.neutral;
                this.ctx.lineWidth = 2;
                this.ctx.beginPath();
                this.ctx.moveTo(50, 250);
                this.ctx.lineTo(750, 250);
                this.ctx.stroke();
                
                // Highlight training period
                this.ctx.fillStyle = this.colors.train + '30';
                this.ctx.fillRect(50, 180, 400, 100);
                
                // Draw patients on timeline
                this.patients.forEach((p, i) => {
                    const x = 80 + (i * 80);
                    const y = 200;
                    const isTraining = i < 5;
                    
                    // Patient box
                    this.ctx.fillStyle = isTraining ? this.colors.train + '40' : '#FFFFFF';
                    this.ctx.fillRect(x - 30, y, 60, 40);
                    this.ctx.strokeStyle = isTraining ? this.colors.train : this.colors.neutral;
                    this.ctx.lineWidth = isTraining ? 2 : 1;
                    this.ctx.strokeRect(x - 30, y, 60, 40);
                    
                    // Patient label
                    this.ctx.fillStyle = this.colors.text;
                    this.ctx.font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto';
                    this.ctx.textAlign = 'center';
                    this.ctx.fillText(p.name.split(' ')[1], x, y + 25);
                    
                    // Timestamp
                    this.ctx.font = '10px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto';
                    this.ctx.fillText(timestamps[i], x, y + 55);
                    
                    // Timeline marker
                    this.ctx.beginPath();
                    this.ctx.arc(x, 250, 4, 0, 2 * Math.PI);
                    this.ctx.fillStyle = isTraining ? this.colors.train : this.colors.primary;
                    this.ctx.fill();
                });
                
                // Label
                this.ctx.fillStyle = this.colors.train;
                this.ctx.font = 'bold 16px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto';
                this.ctx.textAlign = 'center';
                this.ctx.fillText('TRAINING PERIOD', 250, 160);
                
                this.drawMessage('Train on historical data (Jan-May 2023)', 'info');
            },
            () => {
                this.clear();
                this.drawTitle('Temporal Validation - Test on Future');
                
                // Draw timeline
                this.ctx.strokeStyle = this.colors.neutral;
                this.ctx.lineWidth = 2;
                this.ctx.beginPath();
                this.ctx.moveTo(50, 250);
                this.ctx.lineTo(750, 250);
                this.ctx.stroke();
                
                // Highlight periods
                this.ctx.fillStyle = this.colors.train + '30';
                this.ctx.fillRect(50, 180, 400, 100);
                this.ctx.fillStyle = this.colors.test + '30';
                this.ctx.fillRect(450, 180, 280, 100);
                
                // Draw patients on timeline
                this.patients.forEach((p, i) => {
                    const x = 80 + (i * 80);
                    const y = 200;
                    const isTraining = i < 5;
                    
                    // Patient box
                    this.ctx.fillStyle = isTraining ? this.colors.train + '40' : this.colors.test + '40';
                    this.ctx.fillRect(x - 30, y, 60, 40);
                    this.ctx.strokeStyle = isTraining ? this.colors.train : this.colors.test;
                    this.ctx.lineWidth = 2;
                    this.ctx.strokeRect(x - 30, y, 60, 40);
                    
                    // Patient label
                    this.ctx.fillStyle = this.colors.text;
                    this.ctx.font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto';
                    this.ctx.textAlign = 'center';
                    this.ctx.fillText(p.name.split(' ')[1], x, y + 25);
                    
                    // Timestamp
                    this.ctx.font = '10px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto';
                    this.ctx.fillText(timestamps[i], x, y + 55);
                    
                    // Timeline marker
                    this.ctx.beginPath();
                    this.ctx.arc(x, 250, 4, 0, 2 * Math.PI);
                    this.ctx.fillStyle = isTraining ? this.colors.train : this.colors.test;
                    this.ctx.fill();
                });
                
                // Labels
                this.ctx.font = 'bold 16px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto';
                this.ctx.fillStyle = this.colors.train;
                this.ctx.fillText('TRAINING PERIOD', 250, 160);
                this.ctx.fillStyle = this.colors.test;
                this.ctx.fillText('TEST PERIOD', 590, 160);
                
                // Arrow showing time direction
                this.ctx.strokeStyle = this.colors.primary;
                this.ctx.lineWidth = 3;
                this.ctx.beginPath();
                this.ctx.moveTo(430, 250);
                this.ctx.lineTo(470, 250);
                this.ctx.moveTo(460, 240);
                this.ctx.lineTo(470, 250);
                this.ctx.lineTo(460, 260);
                this.ctx.stroke();
                
                this.drawMessage('Test on future data (Jun-Aug 2023) - No look-ahead bias!', 'success');
            }
        ];
        
        if (this.animationStep < steps.length) {
            steps[this.animationStep]();
            this.animationStep++;
        } else {
            this.animationStep = 0;
            this.isAnimating = false;
        }
    }
    
    start(demoType) {
        if (this.isAnimating) return;
        
        this.currentDemo = demoType;
        this.animationStep = 0;
        this.isAnimating = true;
        
        const animate = () => {
            if (!this.isAnimating) return;
            
            switch(this.currentDemo) {
                case 'correct':
                    this.animateCorrectSplitting();
                    break;
                case 'incorrect':
                    this.animateIncorrectSplitting();
                    break;
                case 'kfold':
                    this.animateKFoldCV();
                    break;
                case 'temporal':
                    this.animateTemporalValidation();
                    break;
            }
            
            if (this.isAnimating) {
                setTimeout(() => animate(), 2000);
            }
        };
        
        animate();
    }
    
    stop() {
        this.isAnimating = false;
        this.animationStep = 0;
    }
    
    reset() {
        this.stop();
        this.clear();
        this.initializePatients();
        this.drawTitle('Trustworthy Cross-Validation Demo');
        this.patients.forEach(p => this.drawPatient(p));
        this.drawMessage('Click a demo button to start animation', 'info');
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    const canvas = document.getElementById('cv-animation-canvas');
    if (canvas) {
        window.cvAnimation = new TrustCVAnimation('cv-animation-canvas');
        window.cvAnimation.reset();
    }
});