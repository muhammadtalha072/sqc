-- One-time provisioning. Run as a superuser, once per environment:
--     psql -U postgres -f scripts/bootstrap.sql
--
-- Creates two roles with different privilege levels:
--   sqc      - owns the schema, runs migrations. No CREATEROLE, no superuser.
--   sqc_app  - what the application connects as. Owns nothing, so RLS
--              always applies to it. This separation is the whole point:
--              an application-layer SQL injection cannot disable a policy
--              it does not own.
--
-- Change both passwords before running anywhere real.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'sqc') THEN
        CREATE ROLE sqc LOGIN PASSWORD 'sqc';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'sqc_app') THEN
        CREATE ROLE sqc_app LOGIN PASSWORD 'changeme';
    END IF;
END $$;

-- CREATE DATABASE cannot run inside a transaction block or a DO block, so it
-- is handled by the caller:
--     createdb -O sqc sqc
