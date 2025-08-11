"""
OpenAPI/Swagger Schema Definition
=================================

Defines the OpenAPI specification for the Trading Bot API.
"""

from typing import Dict, Any


def get_openapi_spec() -> Dict[str, Any]:
    """Return OpenAPI 3.0 specification"""
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "Altcoin Trading Bot API",
            "version": "1.0.0",
            "description": "REST API for managing and monitoring the altcoin trading bot",
            "contact": {
                "name": "Trading Bot Support",
                "email": "support@tradingbot.com"
            },
            "license": {
                "name": "MIT",
                "url": "https://opensource.org/licenses/MIT"
            }
        },
        "servers": [
            {
                "url": "http://localhost:5000",
                "description": "Development server"
            },
            {
                "url": "https://api.tradingbot.com",
                "description": "Production server"
            }
        ],
        "components": {
            "securitySchemes": {
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT"
                }
            },
            "schemas": {
                "Error": {
                    "type": "object",
                    "required": ["error", "message"],
                    "properties": {
                        "error": {
                            "type": "string",
                            "description": "Error type"
                        },
                        "message": {
                            "type": "string",
                            "description": "Error message"
                        },
                        "error_id": {
                            "type": "string",
                            "description": "Unique error identifier"
                        },
                        "details": {
                            "type": "object",
                            "description": "Additional error details"
                        }
                    }
                },
                "AuthResponse": {
                    "type": "object",
                    "required": ["access_token", "token_type"],
                    "properties": {
                        "access_token": {
                            "type": "string",
                            "description": "JWT access token"
                        },
                        "refresh_token": {
                            "type": "string",
                            "description": "JWT refresh token"
                        },
                        "token_type": {
                            "type": "string",
                            "enum": ["Bearer"]
                        },
                        "user": {
                            "type": "object",
                            "properties": {
                                "username": {"type": "string"},
                                "roles": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            }
                        }
                    }
                },
                "Position": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Trading symbol"
                        },
                        "amount": {
                            "type": "number",
                            "description": "Position size"
                        },
                        "entry_price": {
                            "type": "number",
                            "description": "Entry price"
                        },
                        "current_price": {
                            "type": "number",
                            "description": "Current market price"
                        },
                        "unrealized_pnl": {
                            "type": "number",
                            "description": "Unrealized profit/loss"
                        },
                        "realized_pnl": {
                            "type": "number",
                            "description": "Realized profit/loss"
                        },
                        "entry_time": {
                            "type": "string",
                            "format": "date-time",
                            "description": "Position entry time"
                        }
                    }
                },
                "Order": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Order ID"
                        },
                        "symbol": {
                            "type": "string",
                            "description": "Trading symbol"
                        },
                        "side": {
                            "type": "string",
                            "enum": ["buy", "sell"]
                        },
                        "type": {
                            "type": "string",
                            "enum": ["market", "limit", "stop"]
                        },
                        "amount": {
                            "type": "number",
                            "description": "Order amount"
                        },
                        "price": {
                            "type": "number",
                            "description": "Order price"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["open", "closed", "cancelled", "partial"]
                        },
                        "filled": {
                            "type": "number",
                            "description": "Filled amount"
                        },
                        "remaining": {
                            "type": "number",
                            "description": "Remaining amount"
                        },
                        "timestamp": {
                            "type": "string",
                            "format": "date-time"
                        }
                    }
                },
                "Strategy": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Strategy name"
                        },
                        "description": {
                            "type": "string",
                            "description": "Strategy description"
                        },
                        "parameters": {
                            "type": "object",
                            "description": "Strategy parameters"
                        },
                        "risk_level": {
                            "type": "string",
                            "enum": ["low", "medium", "high"]
                        },
                        "timeframes": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "markets": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                },
                "Performance": {
                    "type": "object",
                    "properties": {
                        "total_return": {
                            "type": "number",
                            "description": "Total return percentage"
                        },
                        "win_rate": {
                            "type": "number",
                            "description": "Win rate percentage"
                        },
                        "sharpe_ratio": {
                            "type": "number",
                            "description": "Sharpe ratio"
                        },
                        "max_drawdown": {
                            "type": "number",
                            "description": "Maximum drawdown"
                        },
                        "total_trades": {
                            "type": "integer",
                            "description": "Total number of trades"
                        },
                        "avg_trade_duration": {
                            "type": "number",
                            "description": "Average trade duration in hours"
                        }
                    }
                },
                "SystemMetrics": {
                    "type": "object",
                    "properties": {
                        "cpu": {
                            "type": "object",
                            "properties": {
                                "percent": {"type": "number"},
                                "count": {"type": "integer"}
                            }
                        },
                        "memory": {
                            "type": "object",
                            "properties": {
                                "total": {"type": "integer"},
                                "available": {"type": "integer"},
                                "percent": {"type": "number"}
                            }
                        },
                        "disk": {
                            "type": "object",
                            "properties": {
                                "total": {"type": "integer"},
                                "used": {"type": "integer"},
                                "percent": {"type": "number"}
                            }
                        }
                    }
                },
                "Alert": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Alert ID"
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["info", "warning", "error", "critical"]
                        },
                        "message": {
                            "type": "string",
                            "description": "Alert message"
                        },
                        "timestamp": {
                            "type": "string",
                            "format": "date-time"
                        }
                    }
                }
            }
        },
        "paths": {
            "/auth/login": {
                "post": {
                    "tags": ["Authentication"],
                    "summary": "User login",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["username", "password"],
                                    "properties": {
                                        "username": {"type": "string"},
                                        "password": {"type": "string"}
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Login successful",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/AuthResponse"}
                                }
                            }
                        },
                        "401": {
                            "description": "Invalid credentials",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Error"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/v1/trading/status": {
                "get": {
                    "tags": ["Trading"],
                    "summary": "Get trading bot status",
                    "security": [{"BearerAuth": []}],
                    "responses": {
                        "200": {
                            "description": "Trading bot status",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "active": {"type": "boolean"},
                                            "mode": {
                                                "type": "string",
                                                "enum": ["live", "paper", "backtest"]
                                            },
                                            "current_strategy": {"type": "string"},
                                            "positions": {
                                                "type": "array",
                                                "items": {"$ref": "#/components/schemas/Position"}
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "401": {
                            "description": "Authentication required",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Error"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/v1/trading/positions": {
                "get": {
                    "tags": ["Trading"],
                    "summary": "Get all open positions",
                    "security": [{"BearerAuth": []}],
                    "responses": {
                        "200": {
                            "description": "List of open positions",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "positions": {
                                                "type": "array",
                                                "items": {"$ref": "#/components/schemas/Position"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/v1/strategies/list": {
                "get": {
                    "tags": ["Strategies"],
                    "summary": "List all available strategies",
                    "security": [{"BearerAuth": []}],
                    "responses": {
                        "200": {
                            "description": "List of available strategies",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "strategies": {
                                                "type": "array",
                                                "items": {"$ref": "#/components/schemas/Strategy"}
                                            },
                                            "count": {"type": "integer"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/v1/monitoring/health": {
                "get": {
                    "tags": ["Monitoring"],
                    "summary": "System health check",
                    "responses": {
                        "200": {
                            "description": "System is healthy",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {
                                                "type": "string",
                                                "enum": ["healthy", "degraded", "unhealthy"]
                                            },
                                            "timestamp": {
                                                "type": "string",
                                                "format": "date-time"
                                            },
                                            "checks": {
                                                "type": "object",
                                                "additionalProperties": {"type": "string"}
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "503": {
                            "description": "System is unhealthy",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Error"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/v1/monitoring/metrics": {
                "get": {
                    "tags": ["Monitoring"],
                    "summary": "Get system metrics",
                    "security": [{"BearerAuth": []}],
                    "responses": {
                        "200": {
                            "description": "System metrics",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/SystemMetrics"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/v1/monitoring/alerts": {
                "get": {
                    "tags": ["Monitoring"],
                    "summary": "Get active system alerts",
                    "security": [{"BearerAuth": []}],
                    "responses": {
                        "200": {
                            "description": "Active alerts",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "alerts": {
                                                "type": "array",
                                                "items": {"$ref": "#/components/schemas/Alert"}
                                            },
                                            "count": {"type": "integer"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "tags": [
            {
                "name": "Authentication",
                "description": "User authentication and authorization"
            },
            {
                "name": "Trading",
                "description": "Trading bot operations and management"
            },
            {
                "name": "Strategies",
                "description": "Strategy configuration and management"
            },
            {
                "name": "Monitoring",
                "description": "System monitoring and health checks"
            }
        ]
    }