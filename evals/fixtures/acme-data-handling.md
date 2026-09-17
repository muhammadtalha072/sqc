# Acme Corp Data Handling Standard

Version 1.0
Effective Date: 03 March 2024

## 1. Encryption

Customer data is encrypted at rest using AES-256. Keys are held in the cloud
key management service and rotated annually.

## 2. Retention

Customer records are retained for 90 days after account closure, unless a legal
hold applies, in which case records are retained until the hold is lifted.

## 3. Backups

Backups run every four hours. The recovery point objective is four hours.
