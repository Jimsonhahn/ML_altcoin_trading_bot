#!/usr/bin/env python3
"""
ML Altcoin Trading Bot - Database Setup Script
Produktionsreife PostgreSQL/TimescaleDB Initialisierung

Dieses Script:
- Stellt Verbindung zu PostgreSQL/TimescaleDB her
- Erstellt alle Tabellen aus schema.sql
- Richtet Hypertables für Zeitreihen ein
- Legt Indizes für Performance an
- Konfiguriert Connection Pooling
- Implementiert Error Handling und Logging
"""

import os
import sys
import json
import asyncio
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import asyncpg
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2.pool import ThreadedConnectionPool

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('database_setup.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration dataclass"""
    host: str
    port: int
    database: str
    username: str
    password: str
    min_connections: int = 5
    max_connections: int = 20
    ssl_mode: str = 'prefer'
    command_timeout: int = 30
    
    @classmethod
    def from_file(cls, config_path: str) -> 'DatabaseConfig':
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            return cls(
                host=config_data['postgresql']['host'],
                port=config_data['postgresql']['port'],
                database=config_data['postgresql']['database'],
                username=config_data['postgresql']['username'],
                password=config_data['postgresql']['password'],
                min_connections=config_data['postgresql'].get('min_connections', 5),
                max_connections=config_data['postgresql'].get('max_connections', 20),
                ssl_mode=config_data['postgresql'].get('ssl_mode', 'prefer'),
                command_timeout=config_data['postgresql'].get('command_timeout', 30)
            )
        except Exception as e:
            logger.error(f"Failed to load database config: {e}")
            raise

    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Load configuration from environment variables"""
        return cls(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', '5432')),
            database=os.getenv('DB_NAME', 'trading_bot'),
            username=os.getenv('DB_USER', 'trading_bot'),
            password=os.getenv('DB_PASSWORD', ''),
            min_connections=int(os.getenv('DB_MIN_CONNECTIONS', '5')),
            max_connections=int(os.getenv('DB_MAX_CONNECTIONS', '20')),
            ssl_mode=os.getenv('DB_SSL_MODE', 'prefer'),
            command_timeout=int(os.getenv('DB_COMMAND_TIMEOUT', '30'))
        )

class DatabaseSetup:
    """Database setup and management class"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection_pool: Optional[ThreadedConnectionPool] = None
        self.async_pool: Optional[asyncpg.Pool] = None
        
    async def create_database_if_not_exists(self) -> bool:
        """Create database if it doesn't exist"""
        try:
            # Connect to postgres database to create our target database
            conn = await asyncpg.connect(
                host=self.config.host,
                port=self.config.port,
                database='postgres',
                user=self.config.username,
                password=self.config.password,
                ssl=self.config.ssl_mode,
                command_timeout=self.config.command_timeout
            )
            
            # Check if database exists
            result = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1",
                self.config.database
            )
            
            if not result:
                logger.info(f"Creating database '{self.config.database}'...")
                await conn.execute(f'CREATE DATABASE "{self.config.database}"')
                logger.info(f"Database '{self.config.database}' created successfully")
            else:
                logger.info(f"Database '{self.config.database}' already exists")
            
            await conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to create database: {e}")
            return False

    async def setup_extensions(self) -> bool:
        """Setup required PostgreSQL extensions"""
        try:
            conn = await asyncpg.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
                ssl=self.config.ssl_mode,
                command_timeout=self.config.command_timeout
            )
            
            extensions = [
                'timescaledb',
                'uuid-ossp',
                'ltree'
            ]
            
            for ext in extensions:
                try:
                    await conn.execute(f'CREATE EXTENSION IF NOT EXISTS "{ext}" CASCADE')
                    logger.info(f"Extension '{ext}' setup successfully")
                except asyncpg.UndefinedObjectError:
                    logger.warning(f"Extension '{ext}' not available - skipping")
                except Exception as e:
                    logger.error(f"Failed to create extension '{ext}': {e}")
                    
            await conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup extensions: {e}")
            return False

    async def execute_schema_file(self, schema_path: str) -> bool:
        """Execute SQL schema file"""
        try:
            if not os.path.exists(schema_path):
                logger.error(f"Schema file not found: {schema_path}")
                return False
                
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_sql = f.read()
            
            # Hash the schema for version tracking
            schema_hash = hashlib.sha256(schema_sql.encode()).hexdigest()
            logger.info(f"Executing schema (hash: {schema_hash[:16]}...)")
            
            conn = await asyncpg.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
                ssl=self.config.ssl_mode,
                command_timeout=60  # Longer timeout for schema creation
            )
            
            # Split SQL file into individual statements
            statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
            
            success_count = 0
            for i, statement in enumerate(statements):
                try:
                    await conn.execute(statement)
                    success_count += 1
                except Exception as e:
                    # Some statements may fail if they already exist - log but continue
                    if "already exists" in str(e).lower():
                        logger.debug(f"Statement {i+1} skipped (already exists): {e}")
                        success_count += 1
                    else:
                        logger.error(f"Statement {i+1} failed: {e}")
                        logger.debug(f"Failed statement: {statement[:200]}...")
            
            await conn.close()
            
            logger.info(f"Schema execution completed: {success_count}/{len(statements)} statements successful")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Failed to execute schema file: {e}")
            return False

    async def verify_tables(self) -> Dict[str, bool]:
        """Verify that all required tables exist"""
        required_tables = [
            'orchestrator_decisions',
            'strategy_performance', 
            'market_states',
            'ml_insights',
            'strategy_combinations'
        ]
        
        table_status = {}
        
        try:
            conn = await asyncpg.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
                ssl=self.config.ssl_mode,
                command_timeout=self.config.command_timeout
            )
            
            for table in required_tables:
                result = await conn.fetchval(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = $1
                    )
                    """,
                    table
                )
                table_status[table] = bool(result)
                logger.info(f"Table '{table}': {'✓' if result else '✗'}")
            
            await conn.close()
            
        except Exception as e:
            logger.error(f"Failed to verify tables: {e}")
            for table in required_tables:
                table_status[table] = False
                
        return table_status

    async def verify_hypertables(self) -> Dict[str, bool]:
        """Verify TimescaleDB hypertables are properly configured"""
        hypertables = [
            'orchestrator_decisions',
            'strategy_performance',
            'market_states', 
            'ml_insights',
            'strategy_combinations'
        ]
        
        hypertable_status = {}
        
        try:
            conn = await asyncpg.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
                ssl=self.config.ssl_mode,
                command_timeout=self.config.command_timeout
            )
            
            for table in hypertables:
                try:
                    result = await conn.fetchval(
                        """
                        SELECT EXISTS (
                            SELECT 1 FROM timescaledb_information.hypertables 
                            WHERE hypertable_name = $1
                        )
                        """,
                        table
                    )
                    hypertable_status[table] = bool(result)
                    logger.info(f"Hypertable '{table}': {'✓' if result else '✗'}")
                except Exception:
                    # TimescaleDB not available
                    hypertable_status[table] = False
                    logger.debug(f"Hypertable check failed for '{table}' - TimescaleDB not available")
            
            await conn.close()
            
        except Exception as e:
            logger.error(f"Failed to verify hypertables: {e}")
            for table in hypertables:
                hypertable_status[table] = False
                
        return hypertable_status

    def setup_connection_pool(self) -> bool:
        """Setup connection pool for synchronous operations"""
        try:
            self.connection_pool = ThreadedConnectionPool(
                minconn=self.config.min_connections,
                maxconn=self.config.max_connections,
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
                sslmode=self.config.ssl_mode
            )
            
            # Test connection
            conn = self.connection_pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT version()")
                    version = cur.fetchone()[0]
                    logger.info(f"Connection pool established - PostgreSQL version: {version}")
            finally:
                self.connection_pool.putconn(conn)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup connection pool: {e}")
            return False

    async def setup_async_pool(self) -> bool:
        """Setup async connection pool"""
        try:
            self.async_pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
                ssl=self.config.ssl_mode,
                min_size=self.config.min_connections,
                max_size=self.config.max_connections,
                command_timeout=self.config.command_timeout
            )
            
            # Test async pool
            async with self.async_pool.acquire() as conn:
                version = await conn.fetchval("SELECT version()")
                logger.info(f"Async connection pool established - PostgreSQL version: {version}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup async connection pool: {e}")
            return False

    async def create_initial_data(self) -> bool:
        """Create initial data for system tables"""
        try:
            if not self.async_pool:
                logger.error("Async pool not initialized")
                return False
                
            async with self.async_pool.acquire() as conn:
                # Create initial market state entry
                await conn.execute("""
                    INSERT INTO market_states (
                        data_source, market_type, detected_regime, 
                        regime_confidence, total_market_cap, btc_dominance,
                        fear_greedy_index, vix_crypto, realized_volatility_24h,
                        total_volume_24h, trend_direction, anomaly_score
                    ) VALUES (
                        'system', 'spot', 'neutral', 0.5, 
                        1000000000000, 0.45, 50, 0.3, 0.25,
                        50000000000, 'sideways', 0.0
                    )
                    ON CONFLICT DO NOTHING
                """)
                
                logger.info("Initial data created successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create initial data: {e}")
            return False

    async def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check"""
        health_status = {
            'timestamp': datetime.utcnow().isoformat(),
            'database_connection': False,
            'timescaledb_available': False,
            'tables_exist': {},
            'hypertables_configured': {},
            'connection_pools': {
                'sync': False,
                'async': False
            },
            'performance_test': None
        }
        
        try:
            # Test basic connection
            conn = await asyncpg.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
                ssl=self.config.ssl_mode,
                command_timeout=self.config.command_timeout
            )
            
            health_status['database_connection'] = True
            
            # Check TimescaleDB
            try:
                version = await conn.fetchval("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")
                health_status['timescaledb_available'] = version is not None
                if version:
                    logger.info(f"TimescaleDB version: {version}")
            except Exception:
                health_status['timescaledb_available'] = False
            
            await conn.close()
            
            # Verify tables and hypertables
            health_status['tables_exist'] = await self.verify_tables()
            health_status['hypertables_configured'] = await self.verify_hypertables()
            
            # Test connection pools
            health_status['connection_pools']['sync'] = self.connection_pool is not None
            health_status['connection_pools']['async'] = self.async_pool is not None
            
            # Performance test
            start_time = datetime.utcnow()
            if self.async_pool:
                async with self.async_pool.acquire() as conn:
                    await conn.execute("SELECT 1")
            end_time = datetime.utcnow()
            health_status['performance_test'] = {
                'query_latency_ms': (end_time - start_time).total_seconds() * 1000
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
        
        return health_status

    def close_connections(self):
        """Close all connections and pools"""
        try:
            if self.connection_pool:
                self.connection_pool.closeall()
                logger.info("Sync connection pool closed")
                
            # Async pool will be closed by the event loop
            logger.info("Connection cleanup completed")
            
        except Exception as e:
            logger.error(f"Error closing connections: {e}")

    async def close_async_connections(self):
        """Close async connections"""
        try:
            if self.async_pool:
                await self.async_pool.close()
                logger.info("Async connection pool closed")
        except Exception as e:
            logger.error(f"Error closing async connections: {e}")

async def main():
    """Main setup function"""
    logger.info("=== ML Altcoin Trading Bot - Database Setup ===")
    
    # Determine config source
    config_path = "config/database_config.json"
    if os.path.exists(config_path):
        logger.info(f"Loading config from {config_path}")
        config = DatabaseConfig.from_file(config_path)
    else:
        logger.info("Loading config from environment variables")
        config = DatabaseConfig.from_env()
    
    logger.info(f"Connecting to database: {config.host}:{config.port}/{config.database}")
    
    # Initialize database setup
    db_setup = DatabaseSetup(config)
    
    try:
        # Step 1: Create database if needed
        success = await db_setup.create_database_if_not_exists()
        if not success:
            logger.error("Failed to create database")
            return False
        
        # Step 2: Setup extensions
        success = await db_setup.setup_extensions()
        if not success:
            logger.error("Failed to setup extensions")
            return False
        
        # Step 3: Execute schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
        success = await db_setup.execute_schema_file(schema_path)
        if not success:
            logger.error("Failed to execute schema")
            return False
        
        # Step 4: Setup connection pools
        success = db_setup.setup_connection_pool()
        if not success:
            logger.error("Failed to setup sync connection pool")
            return False
            
        success = await db_setup.setup_async_pool()
        if not success:
            logger.error("Failed to setup async connection pool")
            return False
        
        # Step 5: Create initial data
        success = await db_setup.create_initial_data()
        if not success:
            logger.warning("Failed to create initial data - continuing anyway")
        
        # Step 6: Run health check
        health_status = await db_setup.run_health_check()
        
        logger.info("=== Database Setup Health Check ===")
        logger.info(f"Database Connection: {'✓' if health_status['database_connection'] else '✗'}")
        logger.info(f"TimescaleDB Available: {'✓' if health_status['timescaledb_available'] else '✗'}")
        logger.info(f"Sync Pool: {'✓' if health_status['connection_pools']['sync'] else '✗'}")
        logger.info(f"Async Pool: {'✓' if health_status['connection_pools']['async'] else '✗'}")
        
        tables_ok = all(health_status['tables_exist'].values())
        logger.info(f"All Tables Created: {'✓' if tables_ok else '✗'}")
        
        if health_status['performance_test']:
            latency = health_status['performance_test']['query_latency_ms']
            logger.info(f"Query Latency: {latency:.2f}ms")
        
        # Save health check results
        with open('database_health_check.json', 'w') as f:
            json.dump(health_status, f, indent=2, default=str)
        
        if health_status['database_connection'] and tables_ok:
            logger.info("✅ Database setup completed successfully!")
            return True
        else:
            logger.error("❌ Database setup completed with errors")
            return False
            
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        return False
        
    finally:
        # Cleanup
        db_setup.close_connections()
        await db_setup.close_async_connections()

if __name__ == "__main__":
    # Check dependencies
    try:
        import asyncpg
        import psycopg2
    except ImportError as e:
        logger.error(f"Missing required dependency: {e}")
        logger.error("Install with: pip install asyncpg psycopg2-binary")
        sys.exit(1)
    
    # Run setup
    success = asyncio.run(main())
    sys.exit(0 if success else 1)