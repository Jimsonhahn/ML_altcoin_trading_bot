/**
 * 🔊 FACTORY SOUND SYSTEM 🔊
 * Ultimate audio experience for the Janics Freedom Factory!
 */

class FactorySoundSystem {
    constructor() {
        this.enabled = true;
        this.volume = 0.3;
        this.sounds = new Map();
        this.audioContext = null;
        this.masterGain = null;
        this.initializeAudioContext();
        this.createFactorySounds();
    }

    initializeAudioContext() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.masterGain = this.audioContext.createGain();
            this.masterGain.connect(this.audioContext.destination);
            this.masterGain.gain.value = this.volume;
        } catch (error) {
            console.warn('Audio context not supported:', error);
            this.enabled = false;
        }
    }

    createFactorySounds() {
        // Money printer sounds
        this.createToneSound('moneyPrint', [400, 600, 800], 0.3, 'sine');
        this.createToneSound('coinDrop', [800, 400], 0.2, 'triangle');
        this.createToneSound('cashRegister', [523, 659, 784, 1047], 0.4, 'square');
        
        // Factory operation sounds
        this.createToneSound('factoryHum', [100, 150, 200], 2, 'sawtooth');
        this.createToneSound('machineStart', [200, 400, 600], 1, 'sine');
        this.createToneSound('steamRelease', [800, 200], 0.8, 'square');
        
        // Trading sounds
        this.createToneSound('tradeOpen', [440, 554, 659], 0.5, 'sine');
        this.createToneSound('tradeClose', [659, 554, 440], 0.5, 'sine');
        this.createToneSound('profitBell', [880, 1100, 1320], 0.7, 'triangle');
        this.createToneSound('lossAlert', [220, 185, 147], 0.6, 'sawtooth');
        
        // UI interaction sounds
        this.createToneSound('buttonClick', [800, 1000], 0.1, 'square');
        this.createToneSound('buttonHover', [1200], 0.05, 'sine');
        this.createToneSound('success', [523, 659, 784], 0.4, 'triangle');
        this.createToneSound('error', [200, 150, 100], 0.5, 'sawtooth');
        this.createToneSound('warning', [300, 350, 300], 0.3, 'square');
        
        // AI and brain sounds
        this.createToneSound('brainPulse', [400, 500, 600], 0.5, 'sine');
        this.createToneSound('dataProcess', [800, 1000, 1200, 1400], 0.3, 'triangle');
        this.createToneSound('neuralFire', [1500, 1800], 0.2, 'sine');
        
        // Emergency sounds
        this.createToneSound('emergency', [440, 880, 440, 880], 2, 'sawtooth');
        this.createToneSound('alarm', [800, 400, 800, 400], 1.5, 'square');
    }

    createToneSound(name, frequencies, duration, waveType = 'sine') {
        this.sounds.set(name, {
            frequencies,
            duration,
            waveType,
            type: 'tone'
        });
    }

    playSound(soundName, options = {}) {
        if (!this.enabled || !this.audioContext || !this.sounds.has(soundName)) {
            return;
        }

        // Resume audio context if suspended (required by browsers)
        if (this.audioContext.state === 'suspended') {
            this.audioContext.resume();
        }

        const sound = this.sounds.get(soundName);
        const { volume = 1, pitch = 1, delay = 0 } = options;

        setTimeout(() => {
            this.playToneSequence(sound, volume, pitch);
        }, delay * 1000);
    }

    playToneSequence(sound, volumeMultiplier = 1, pitchMultiplier = 1) {
        const { frequencies, duration, waveType } = sound;
        const toneDuration = duration / frequencies.length;
        const currentTime = this.audioContext.currentTime;

        frequencies.forEach((frequency, index) => {
            const oscillator = this.audioContext.createOscillator();
            const gainNode = this.audioContext.createGain();
            
            oscillator.connect(gainNode);
            gainNode.connect(this.masterGain);
            
            oscillator.type = waveType;
            oscillator.frequency.setValueAtTime(
                frequency * pitchMultiplier, 
                currentTime + (index * toneDuration)
            );
            
            // Envelope for smoother sound
            const startTime = currentTime + (index * toneDuration);
            const endTime = startTime + toneDuration;
            
            gainNode.gain.setValueAtTime(0, startTime);
            gainNode.gain.linearRampToValueAtTime(
                this.volume * volumeMultiplier * 0.1, 
                startTime + 0.01
            );
            gainNode.gain.exponentialRampToValueAtTime(
                0.001, 
                endTime - 0.01
            );
            
            oscillator.start(startTime);
            oscillator.stop(endTime);
        });
    }

    // Specialized sound effects
    playMoneySound(amount) {
        const intensity = Math.min(amount / 1000, 1);
        
        if (amount > 500) {
            this.playSound('cashRegister', { volume: intensity });
        } else if (amount > 100) {
            this.playSound('coinDrop', { volume: intensity });
        } else {
            this.playSound('moneyPrint', { volume: intensity * 0.5 });
        }
    }

    playTradeSound(type, profit = 0) {
        switch (type) {
            case 'open':
                this.playSound('tradeOpen');
                break;
            case 'close':
                this.playSound('tradeClose');
                if (profit > 0) {
                    setTimeout(() => this.playSound('profitBell'), 300);
                } else if (profit < 0) {
                    setTimeout(() => this.playSound('lossAlert'), 300);
                }
                break;
        }
    }

    playBotStatusSound(status) {
        switch (status) {
            case 'starting':
                this.playSound('machineStart');
                break;
            case 'running':
                this.playSound('factoryHum', { volume: 0.3 });
                break;
            case 'stopped':
                this.playSound('steamRelease');
                break;
            case 'error':
                this.playSound('error');
                break;
            case 'emergency':
                this.playSound('emergency');
                break;
        }
    }

    playUISound(interaction) {
        switch (interaction) {
            case 'click':
                this.playSound('buttonClick');
                break;
            case 'hover':
                this.playSound('buttonHover');
                break;
            case 'success':
                this.playSound('success');
                break;
            case 'error':
                this.playSound('error');
                break;
            case 'warning':
                this.playSound('warning');
                break;
        }
    }

    playBrainActivity() {
        this.playSound('brainPulse', { volume: 0.4 });
        setTimeout(() => this.playSound('dataProcess', { volume: 0.3 }), 200);
        setTimeout(() => this.playSound('neuralFire', { volume: 0.2 }), 400);
    }

    // Sound system controls
    setVolume(volume) {
        this.volume = Math.max(0, Math.min(1, volume));
        if (this.masterGain) {
            this.masterGain.gain.value = this.volume;
        }
    }

    mute() {
        this.enabled = false;
    }

    unmute() {
        this.enabled = true;
    }

    toggle() {
        this.enabled = !this.enabled;
    }

    isEnabled() {
        return this.enabled;
    }

    // Ambient factory sounds
    startAmbientSound() {
        if (!this.enabled) return;
        
        // Subtle factory hum
        setInterval(() => {
            if (this.enabled) {
                this.playSound('factoryHum', { volume: 0.1 });
            }
        }, 10000);
    }
}

// Create singleton instance
const factorySoundSystem = new FactorySoundSystem();

export default factorySoundSystem;