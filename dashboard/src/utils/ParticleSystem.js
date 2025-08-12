/**
 * ✨ ADVANCED PARTICLE SYSTEM ✨
 * Ultimate visual effects for the Janics Freedom Factory!
 */

export class ParticleSystem {
    constructor(container) {
        this.container = container;
        this.particles = [];
        this.animationId = null;
        this.isRunning = false;
        this.effects = new Map();
        
        this.init();
    }

    init() {
        // Create canvas for advanced particle effects
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.canvas.style.position = 'absolute';
        this.canvas.style.top = '0';
        this.canvas.style.left = '0';
        this.canvas.style.pointerEvents = 'none';
        this.canvas.style.zIndex = '10';
        
        this.updateCanvasSize();
        window.addEventListener('resize', () => this.updateCanvasSize());
        
        if (this.container) {
            this.container.appendChild(this.canvas);
        }
    }

    updateCanvasSize() {
        const rect = this.container?.getBoundingClientRect() || { width: window.innerWidth, height: window.innerHeight };
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
    }

    start() {
        if (this.isRunning) return;
        this.isRunning = true;
        this.animate();
    }

    stop() {
        this.isRunning = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
    }

    animate() {
        if (!this.isRunning) return;
        
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Update and draw particles
        for (let i = this.particles.length - 1; i >= 0; i--) {
            const particle = this.particles[i];
            this.updateParticle(particle);
            this.drawParticle(particle);
            
            // Remove dead particles
            if (particle.life <= 0) {
                this.particles.splice(i, 1);
            }
        }
        
        this.animationId = requestAnimationFrame(() => this.animate());
    }

    updateParticle(particle) {
        particle.x += particle.vx;
        particle.y += particle.vy;
        particle.vx *= particle.friction;
        particle.vy *= particle.friction;
        particle.vy += particle.gravity;
        particle.life -= particle.decay;
        particle.size *= particle.sizeDecay;
        particle.rotation += particle.rotationSpeed;
        
        // Apply specific particle behaviors
        if (particle.behavior) {
            particle.behavior(particle);
        }
    }

    drawParticle(particle) {
        this.ctx.save();
        
        this.ctx.globalAlpha = particle.life / particle.maxLife;
        this.ctx.translate(particle.x, particle.y);
        this.ctx.rotate(particle.rotation);
        
        switch (particle.type) {
            case 'circle':
                this.drawCircle(particle);
                break;
            case 'spark':
                this.drawSpark(particle);
                break;
            case 'glow':
                this.drawGlow(particle);
                break;
            case 'text':
                this.drawText(particle);
                break;
            case 'trail':
                this.drawTrail(particle);
                break;
        }
        
        this.ctx.restore();
    }

    drawCircle(particle) {
        this.ctx.beginPath();
        this.ctx.arc(0, 0, particle.size, 0, Math.PI * 2);
        this.ctx.fillStyle = particle.color;
        this.ctx.fill();
        
        if (particle.stroke) {
            this.ctx.strokeStyle = particle.strokeColor;
            this.ctx.lineWidth = particle.strokeWidth;
            this.ctx.stroke();
        }
    }

    drawSpark(particle) {
        this.ctx.beginPath();
        this.ctx.moveTo(-particle.size, 0);
        this.ctx.lineTo(particle.size, 0);
        this.ctx.strokeStyle = particle.color;
        this.ctx.lineWidth = 2;
        this.ctx.stroke();
        
        this.ctx.beginPath();
        this.ctx.moveTo(0, -particle.size);
        this.ctx.lineTo(0, particle.size);
        this.ctx.stroke();
    }

    drawGlow(particle) {
        const gradient = this.ctx.createRadialGradient(0, 0, 0, 0, 0, particle.size);
        gradient.addColorStop(0, particle.color);
        gradient.addColorStop(1, 'transparent');
        
        this.ctx.beginPath();
        this.ctx.arc(0, 0, particle.size, 0, Math.PI * 2);
        this.ctx.fillStyle = gradient;
        this.ctx.fill();
    }

    drawText(particle) {
        this.ctx.font = `${particle.size}px Arial`;
        this.ctx.fillStyle = particle.color;
        this.ctx.textAlign = 'center';
        this.ctx.fillText(particle.text, 0, 0);
    }

    drawTrail(particle) {
        if (particle.trail && particle.trail.length > 1) {
            this.ctx.beginPath();
            this.ctx.moveTo(particle.trail[0].x - particle.x, particle.trail[0].y - particle.y);
            
            for (let i = 1; i < particle.trail.length; i++) {
                this.ctx.lineTo(particle.trail[i].x - particle.x, particle.trail[i].y - particle.y);
            }
            
            this.ctx.strokeStyle = particle.color;
            this.ctx.lineWidth = particle.size;
            this.ctx.stroke();
        }
    }

    // Particle creation methods
    createParticle(config) {
        const defaultConfig = {
            x: 0,
            y: 0,
            vx: 0,
            vy: 0,
            size: 5,
            color: '#ffffff',
            life: 1,
            maxLife: 1,
            decay: 0.01,
            friction: 0.98,
            gravity: 0,
            rotation: 0,
            rotationSpeed: 0,
            sizeDecay: 0.99,
            type: 'circle'
        };

        const particle = { ...defaultConfig, ...config };
        particle.maxLife = particle.life;
        this.particles.push(particle);
        return particle;
    }

    // Pre-defined effects
    createMoneyExplosion(x, y, amount = 100) {
        const particleCount = Math.min(Math.floor(amount / 10), 50);
        
        for (let i = 0; i < particleCount; i++) {
            const angle = (Math.PI * 2 * i) / particleCount;
            const speed = Math.random() * 5 + 3;
            
            this.createParticle({
                x: x,
                y: y,
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed - 2,
                size: Math.random() * 8 + 4,
                color: `hsl(${Math.random() * 60 + 40}, 100%, 60%)`, // Gold colors
                life: Math.random() * 2 + 1,
                decay: 0.02,
                gravity: 0.1,
                rotationSpeed: (Math.random() - 0.5) * 0.2,
                type: 'glow'
            });
        }
    }

    createProfitCelebration(x, y, profit) {
        const symbols = ['💰', '💎', '🤑', '💵', '💸'];
        
        for (let i = 0; i < 10; i++) {
            this.createParticle({
                x: x + (Math.random() - 0.5) * 100,
                y: y + (Math.random() - 0.5) * 50,
                vx: (Math.random() - 0.5) * 4,
                vy: Math.random() * -3 - 2,
                size: Math.random() * 20 + 15,
                color: '#00ff88',
                life: Math.random() * 3 + 2,
                decay: 0.015,
                gravity: 0.05,
                text: symbols[Math.floor(Math.random() * symbols.length)],
                type: 'text'
            });
        }
    }

    createBrainActivity(x, y) {
        for (let i = 0; i < 15; i++) {
            const angle = Math.random() * Math.PI * 2;
            const distance = Math.random() * 30 + 10;
            
            this.createParticle({
                x: x + Math.cos(angle) * distance,
                y: y + Math.sin(angle) * distance,
                vx: (Math.random() - 0.5) * 2,
                vy: (Math.random() - 0.5) * 2,
                size: Math.random() * 4 + 2,
                color: `hsl(240, 100%, ${Math.random() * 50 + 50}%)`,
                life: Math.random() * 2 + 1,
                decay: 0.02,
                type: 'spark'
            });
        }
    }

    createSteamEffect(x, y) {
        for (let i = 0; i < 5; i++) {
            this.createParticle({
                x: x + (Math.random() - 0.5) * 20,
                y: y,
                vx: (Math.random() - 0.5) * 1,
                vy: Math.random() * -2 - 1,
                size: Math.random() * 15 + 10,
                color: `rgba(255, 255, 255, ${Math.random() * 0.5 + 0.2})`,
                life: Math.random() * 3 + 2,
                decay: 0.01,
                sizeDecay: 1.02,
                type: 'circle'
            });
        }
    }

    createSuccessRipple(x, y) {
        for (let i = 0; i < 3; i++) {
            setTimeout(() => {
                this.createParticle({
                    x: x,
                    y: y,
                    vx: 0,
                    vy: 0,
                    size: 5,
                    color: 'rgba(0, 255, 136, 0.3)',
                    life: 1,
                    decay: 0.02,
                    sizeDecay: 1.1,
                    type: 'circle',
                    stroke: true,
                    strokeColor: '#00ff88',
                    strokeWidth: 2
                });
            }, i * 200);
        }
    }

    createErrorShockwave(x, y) {
        this.createParticle({
            x: x,
            y: y,
            vx: 0,
            vy: 0,
            size: 10,
            color: 'rgba(255, 71, 87, 0.2)',
            life: 0.8,
            decay: 0.03,
            sizeDecay: 1.15,
            type: 'circle',
            stroke: true,
            strokeColor: '#ff4757',
            strokeWidth: 3
        });
    }

    createFactorySmoke(x, y) {
        setInterval(() => {
            if (Math.random() < 0.3) { // 30% chance each frame
                this.createParticle({
                    x: x + (Math.random() - 0.5) * 40,
                    y: y,
                    vx: (Math.random() - 0.5) * 0.5,
                    vy: Math.random() * -1 - 0.5,
                    size: Math.random() * 20 + 15,
                    color: `rgba(200, 200, 200, ${Math.random() * 0.3 + 0.1})`,
                    life: Math.random() * 4 + 3,
                    decay: 0.005,
                    sizeDecay: 1.01,
                    type: 'circle'
                });
            }
        }, 500);
    }

    createTradingSignal(x, y, type) {
        const colors = {
            buy: '#00ff88',
            sell: '#ff4757',
            hold: '#ffa500'
        };
        
        const symbols = {
            buy: '📈',
            sell: '📉',
            hold: '➡️'
        };
        
        this.createParticle({
            x: x,
            y: y,
            vx: 0,
            vy: -2,
            size: 30,
            color: colors[type],
            life: 2,
            decay: 0.02,
            text: symbols[type],
            type: 'text',
            behavior: (particle) => {
                particle.size = 30 + Math.sin(particle.maxLife - particle.life) * 5;
            }
        });
    }

    // Continuous effects
    startMoneyRain(intensity = 0.3) {
        this.effects.set('moneyRain', setInterval(() => {
            if (Math.random() < intensity) {
                this.createParticle({
                    x: Math.random() * this.canvas.width,
                    y: -20,
                    vx: (Math.random() - 0.5) * 2,
                    vy: Math.random() * 3 + 2,
                    size: Math.random() * 15 + 10,
                    color: '#ffd700',
                    life: 5,
                    decay: 0.01,
                    rotationSpeed: (Math.random() - 0.5) * 0.1,
                    text: ['💰', '💎', '🪙'][Math.floor(Math.random() * 3)],
                    type: 'text'
                });
            }
        }, 200));
    }

    stopMoneyRain() {
        const effect = this.effects.get('moneyRain');
        if (effect) {
            clearInterval(effect);
            this.effects.delete('moneyRain');
        }
    }

    startElectricCurrent() {
        this.effects.set('electric', setInterval(() => {
            const x = Math.random() * this.canvas.width;
            const y = Math.random() * this.canvas.height;
            
            for (let i = 0; i < 3; i++) {
                this.createParticle({
                    x: x,
                    y: y,
                    vx: (Math.random() - 0.5) * 10,
                    vy: (Math.random() - 0.5) * 10,
                    size: Math.random() * 3 + 1,
                    color: `hsl(${Math.random() * 60 + 180}, 100%, 80%)`,
                    life: 0.5,
                    decay: 0.05,
                    type: 'spark'
                });
            }
        }, 100));
    }

    stopElectricCurrent() {
        const effect = this.effects.get('electric');
        if (effect) {
            clearInterval(effect);
            this.effects.delete('electric');
        }
    }

    // Clean up
    destroy() {
        this.stop();
        this.effects.forEach(effect => clearInterval(effect));
        this.effects.clear();
        if (this.canvas && this.canvas.parentNode) {
            this.canvas.parentNode.removeChild(this.canvas);
        }
    }
}

export default ParticleSystem;