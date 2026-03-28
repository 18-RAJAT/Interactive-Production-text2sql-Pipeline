"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  MessageSquare, Plus, PanelLeftClose, PanelLeft,
  Database, Copy, Check, Clock, Trash2,
  ChevronRight, Sparkles, AlertCircle, Wifi, WifiOff,
  BookOpen, Zap,
} from "lucide-react";
import ClaudeChatInput from "@/components/ui/claude-style-chat-input";
import type { AttachedFile } from "@/components/ui/claude-style-chat-input";
import { generateSQL, checkHealth } from "@/lib/api";
import { cn } from "@/lib/cn";

type MessageType = "sql" | "explanation" | "error";

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  type?: MessageType;
  sql?: string;
  exampleSql?: string;
  confidence?: number;
  latency?: number;
  timestamp: number;
}

interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  schema: string;
  createdAt: number;
  updatedAt: number;
}

const STORAGE_KEY = "text-to-sql-conversations";
const ACTIVE_KEY = "text-to-sql-active-conv";

const MODELS = [
  { id: "lora-finetuned", name: "LoRA Fine-Tuned", description: "Schema-aware SQL generation" },
  { id: "base-model", name: "Base Model", description: "Standard text-to-SQL" },
  { id: "rule-engine", name: "Rule Engine", description: "Heuristic-based generation" },
];

const DEFAULT_SCHEMA = `CREATE TABLE employees (
  id INT PRIMARY KEY,
  name VARCHAR(100),
  department VARCHAR(50),
  salary DECIMAL(10,2),
  hire_date DATE
);

CREATE TABLE departments (
  id INT PRIMARY KEY,
  name VARCHAR(50),
  manager_id INT
);`;

const SQL_GENERATION_PATTERNS = [
  /^(show|find|get|list|fetch|retrieve|display|select|count|calculate|compute)\b/i,
  /^(how many|what('s| is| are) the (total|average|max|min|sum|count))/i,
  /^(give me|return|pull|extract)\b/i,
  /\b(where|from|group by|order by|having|limit|join)\b.*\b(table|column|row|record)\b/i,
  /\b(employees?|departments?|managers?|salary|salaries|hire.?date)\b/i,
  /\b(greater than|less than|more than|above|below|between|equal to)\b.*\b\d+/i,
  /\b(top|bottom|highest|lowest|maximum|minimum|biggest|smallest)\s+\d*/i,
  /\b(last|past|recent|first)\s+\d+\s+(days?|months?|years?|weeks?)/i,
  /\b(sorted|ordered|grouped|filtered)\b/i,
];

const SQL_KNOWLEDGE: Record<string, { title: string; content: string; example: string }> = {
  "what is sql": {
    title: "What is SQL?",
    content: "SQL (Structured Query Language) is a standard programming language designed for managing and manipulating relational databases. It allows you to create, read, update, and delete data stored in tables. SQL is used by virtually every application that works with structured data.\n\nSQL is divided into sub-languages:\n\n• DDL (Data Definition Language) — CREATE, ALTER, DROP, TRUNCATE\n• DML (Data Manipulation Language) — SELECT, INSERT, UPDATE, DELETE\n• DCL (Data Control Language) — GRANT, REVOKE\n• TCL (Transaction Control Language) — COMMIT, ROLLBACK, SAVEPOINT",
    example: "SELECT name, salary\nFROM employees\nWHERE department = 'Engineering'\nORDER BY salary DESC;",
  },
  "ddl dml dcl tcl": {
    title: "DDL / DML / DCL / TCL — SQL Sub-Languages",
    content: "SQL commands are grouped into four categories based on their function:\n\nDDL (Data Definition Language) defines and modifies database structure:\n\n• CREATE — create tables, views, indexes, databases\n• ALTER — modify existing structures (add/drop columns, change types)\n• DROP — permanently remove objects\n• TRUNCATE — remove all rows from a table (faster than DELETE)\n• RENAME — rename an object\n\nDML (Data Manipulation Language) works with the data inside tables:\n\n• SELECT — retrieve/query data\n• INSERT — add new rows\n• UPDATE — modify existing rows\n• DELETE — remove specific rows\n• MERGE — upsert (insert or update)\n\nDCL (Data Control Language) manages permissions and access:\n\n• GRANT — give privileges to users/roles\n• REVOKE — remove privileges from users/roles\n\nTCL (Transaction Control Language) manages transactions:\n\n• COMMIT — save all changes permanently\n• ROLLBACK — undo changes since last commit\n• SAVEPOINT — create a restore point within a transaction\n• SET TRANSACTION — set transaction properties (isolation level, read/write)",
    example: "-- DDL\nCREATE TABLE products (\n  id INT PRIMARY KEY,\n  name VARCHAR(100),\n  price DECIMAL(10,2)\n);\n\n-- DML\nINSERT INTO products VALUES (1, 'Widget', 29.99);\nSELECT * FROM products WHERE price > 20;\n\n-- DCL\nGRANT SELECT, INSERT ON products TO analyst_role;\n\n-- TCL\nBEGIN;\nUPDATE products SET price = 24.99 WHERE id = 1;\nCOMMIT;",
  },
  "what is a join": {
    title: "SQL JOINs",
    content: "A JOIN clause combines rows from two or more tables based on a related column between them. JOINs are fundamental to relational databases because data is typically spread across multiple tables.\n\nTypes of JOINs:\n\n• INNER JOIN — returns only matching rows from both tables\n• LEFT JOIN — returns all rows from the left table, with matches from the right\n• RIGHT JOIN — returns all rows from the right table, with matches from the left\n• FULL OUTER JOIN — returns all rows when there's a match in either table\n• CROSS JOIN — returns the Cartesian product of both tables\n• SELF JOIN — a table joined with itself",
    example: "SELECT e.name, d.name AS department\nFROM employees e\nINNER JOIN departments d\n  ON e.department_id = d.id;",
  },
  "what is a where clause": {
    title: "WHERE Clause",
    content: "The WHERE clause filters rows based on specified conditions. It's used with SELECT, UPDATE, and DELETE statements to target specific records. You can combine multiple conditions using AND, OR, and NOT operators.\n\nCommon operators:\n\n• = , != , <> — equality / inequality\n• < , > , <= , >= — comparison\n• BETWEEN — range check\n• LIKE — pattern matching (% for any chars, _ for single char)\n• IN — match against a list\n• IS NULL / IS NOT NULL — null checks\n• EXISTS — check if a subquery returns rows",
    example: "SELECT name, salary\nFROM employees\nWHERE department = 'Sales'\n  AND salary > 50000\n  AND hire_date BETWEEN '2024-01-01' AND '2024-12-31';",
  },
  "what is group by": {
    title: "GROUP BY Clause",
    content: "GROUP BY groups rows that share values in specified columns into summary rows. It's almost always used with aggregate functions like COUNT, SUM, AVG, MAX, and MIN to compute statistics per group.\n\nThe HAVING clause can filter groups after aggregation (unlike WHERE, which filters rows before grouping).",
    example: "SELECT department, \n       COUNT(*) AS total_employees,\n       AVG(salary) AS avg_salary\nFROM employees\nGROUP BY department\nHAVING COUNT(*) > 5\nORDER BY avg_salary DESC;",
  },
  "what is an index": {
    title: "Database Indexes",
    content: "An index is a data structure that improves the speed of data retrieval on a table at the cost of additional storage and slower writes. Think of it like a book's index — instead of scanning every page, you look up the topic and jump directly to the right page.\n\nWhen to use indexes:\n\n• Columns frequently used in WHERE clauses\n• Columns used in JOIN conditions\n• Columns used in ORDER BY\n\nWhen to avoid:\n\n• Small tables (full scan is fast enough)\n• Columns with many NULL values\n• Tables with heavy INSERT/UPDATE operations",
    example: "CREATE INDEX idx_employee_dept\n  ON employees(department);\n\nCREATE UNIQUE INDEX idx_employee_email\n  ON employees(email);",
  },
  "what is a primary key": {
    title: "Primary Keys & Foreign Keys",
    content: "A PRIMARY KEY uniquely identifies each record in a table. It must contain unique, non-null values. Each table can have only one primary key, though it can span multiple columns (composite key).\n\nA FOREIGN KEY is a column that references the primary key of another table, creating a relationship between the two tables. It enforces referential integrity — you can't insert a value that doesn't exist in the referenced table.",
    example: "CREATE TABLE departments (\n  id INT PRIMARY KEY,\n  name VARCHAR(50)\n);\n\nCREATE TABLE employees (\n  id INT PRIMARY KEY,\n  name VARCHAR(100),\n  dept_id INT,\n  FOREIGN KEY (dept_id) REFERENCES departments(id)\n);",
  },
  "what is a subquery": {
    title: "Subqueries",
    content: "A subquery is a query nested inside another query. It can appear in SELECT, FROM, or WHERE clauses. Subqueries are useful when you need to use the result of one query as input for another.\n\nTypes:\n\n• Scalar subquery — returns a single value\n• Row subquery — returns a single row\n• Table subquery — returns a result set\n• Correlated subquery — references the outer query",
    example: "SELECT name, salary\nFROM employees\nWHERE salary > (\n  SELECT AVG(salary)\n  FROM employees\n);",
  },
  "what is normalization": {
    title: "Database Normalization",
    content: "Normalization is the process of organizing database tables to minimize data redundancy and dependency. It divides large tables into smaller ones and links them using relationships.\n\nNormal Forms:\n\n• 1NF — eliminate repeating groups; each cell holds a single value\n• 2NF — remove partial dependencies (all non-key columns depend on the full primary key)\n• 3NF — remove transitive dependencies (non-key columns depend only on the primary key)\n• BCNF — every determinant is a candidate key",
    example: "-- Instead of one denormalized table:\n-- employees(id, name, dept_name, dept_location)\n\n-- Normalize into two tables:\nCREATE TABLE departments (\n  id INT PRIMARY KEY,\n  name VARCHAR(50),\n  location VARCHAR(100)\n);\n\nCREATE TABLE employees (\n  id INT PRIMARY KEY,\n  name VARCHAR(100),\n  dept_id INT REFERENCES departments(id)\n);",
  },
  "what are aggregate functions": {
    title: "Aggregate Functions",
    content: "Aggregate functions perform calculations on a set of values and return a single result. They're commonly used with GROUP BY to compute statistics per group.\n\nCommon aggregate functions:\n\n• COUNT() — number of rows\n• SUM() — total of a numeric column\n• AVG() — average value\n• MAX() — highest value\n• MIN() — lowest value\n• GROUP_CONCAT() / STRING_AGG() — concatenate values",
    example: "SELECT \n  department,\n  COUNT(*) AS headcount,\n  SUM(salary) AS total_payroll,\n  AVG(salary) AS avg_salary,\n  MAX(salary) AS top_salary,\n  MIN(salary) AS entry_salary\nFROM employees\nGROUP BY department;",
  },
  "difference between where and having": {
    title: "WHERE vs HAVING",
    content: "Both WHERE and HAVING filter data, but they operate at different stages of query execution:\n\nWHERE filters individual rows before any grouping occurs. It cannot use aggregate functions.\n\nHAVING filters groups after GROUP BY has been applied. It can use aggregate functions.\n\nExecution order: FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY",
    example: "SELECT department, AVG(salary) AS avg_sal\nFROM employees\nWHERE hire_date > '2023-01-01'\nGROUP BY department\nHAVING AVG(salary) > 60000;",
  },
  "what is a transaction": {
    title: "Database Transactions",
    content: "A transaction is a sequence of operations performed as a single logical unit of work. Transactions follow the ACID properties:\n\n• Atomicity — all operations succeed or all fail\n• Consistency — database stays in a valid state\n• Isolation — concurrent transactions don't interfere\n• Durability — committed changes survive system failures\n\nKey commands: BEGIN, COMMIT, ROLLBACK, SAVEPOINT",
    example: "BEGIN TRANSACTION;\n\nUPDATE accounts SET balance = balance - 500\n  WHERE id = 1;\n\nUPDATE accounts SET balance = balance + 500\n  WHERE id = 2;\n\nCOMMIT;",
  },
  "what is a view": {
    title: "SQL Views",
    content: "A view is a virtual table based on the result of a SELECT query. It doesn't store data itself — it dynamically pulls data from the underlying tables each time it's queried. Views simplify complex queries, provide a layer of security, and present data in a specific format.\n\nTypes:\n\n• Regular view — a saved SELECT query\n• Materialized view — stores the result set physically (faster reads, needs refresh)\n• Updatable view — allows INSERT/UPDATE/DELETE on simple views",
    example: "CREATE VIEW high_earners AS\nSELECT name, department, salary\nFROM employees\nWHERE salary > 80000;\n\nSELECT * FROM high_earners\nORDER BY salary DESC;",
  },
  "what are window functions": {
    title: "Window Functions",
    content: "Window functions perform calculations across a set of rows that are related to the current row, without collapsing them into a single output row like GROUP BY does. They use the OVER() clause to define the window (partition and ordering).\n\nCommon window functions:\n\n• ROW_NUMBER() — sequential row number within partition\n• RANK() — rank with gaps for ties\n• DENSE_RANK() — rank without gaps\n• LAG() / LEAD() — access previous/next row values\n• SUM() OVER() / AVG() OVER() — running totals/averages\n• NTILE(n) — divide rows into n roughly equal groups",
    example: "SELECT \n  name,\n  department,\n  salary,\n  RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS dept_rank,\n  AVG(salary) OVER (PARTITION BY department) AS dept_avg\nFROM employees;",
  },
  "what is union": {
    title: "UNION & Set Operations",
    content: "UNION combines the result sets of two or more SELECT statements into a single result. The queries must have the same number of columns with compatible data types.\n\nSet operations:\n\n• UNION — combines results, removes duplicates\n• UNION ALL — combines results, keeps duplicates (faster)\n• INTERSECT — returns only rows present in both results\n• EXCEPT / MINUS — returns rows from first result not in second",
    example: "SELECT name, 'Employee' AS type FROM employees\nUNION ALL\nSELECT name, 'Manager' AS type FROM departments\n  INNER JOIN employees ON departments.manager_id = employees.id;",
  },
  "what is order by": {
    title: "ORDER BY Clause",
    content: "ORDER BY sorts the result set by one or more columns. By default, sorting is ascending (ASC). Use DESC for descending order. You can sort by column name, alias, position number, or expression.\n\nNULL handling varies by database:\n\n• PostgreSQL: NULLS FIRST / NULLS LAST\n• MySQL: NULLs sort as lowest values\n• SQL Server: NULLs sort as lowest values",
    example: "SELECT name, department, salary\nFROM employees\nORDER BY department ASC, salary DESC\nLIMIT 20 OFFSET 0;",
  },
  "what is distinct": {
    title: "DISTINCT Keyword",
    content: "DISTINCT eliminates duplicate rows from the result set. It applies to the entire row, not just a single column. For removing duplicates based on specific columns while keeping other data, use GROUP BY or window functions instead.\n\nVariations:\n\n• SELECT DISTINCT — unique rows\n• COUNT(DISTINCT col) — count unique values\n• DISTINCT ON (col) — first row per group (PostgreSQL)",
    example: "SELECT DISTINCT department\nFROM employees\nORDER BY department;\n\nSELECT COUNT(DISTINCT department) AS unique_depts\nFROM employees;",
  },
  "what is null": {
    title: "NULL Handling in SQL",
    content: "NULL represents the absence of a value — it's not zero, not an empty string, and not false. Any comparison with NULL returns NULL (not true or false), which is why you must use IS NULL / IS NOT NULL instead of = NULL.\n\nKey behaviors:\n\n• NULL = NULL → NULL (not true!)\n• NULL + 5 → NULL\n• NULL in WHERE → row is excluded\n• COUNT(*) counts NULLs, COUNT(col) does not\n• Use COALESCE(col, default) to replace NULLs\n• Use NULLIF(a, b) to return NULL when a = b",
    example: "SELECT \n  name,\n  COALESCE(department, 'Unassigned') AS department,\n  COALESCE(salary, 0) AS salary\nFROM employees\nWHERE manager_id IS NOT NULL;",
  },
  "what is case when": {
    title: "CASE / WHEN Expression",
    content: "CASE is SQL's conditional expression (like if/else). It can be used in SELECT, WHERE, ORDER BY, and even inside aggregate functions. There are two forms:\n\nSimple CASE — compares a value against a list:\n\n• CASE col WHEN val1 THEN result1 WHEN val2 THEN result2 END\n\nSearched CASE — evaluates boolean conditions:\n\n• CASE WHEN condition1 THEN result1 WHEN condition2 THEN result2 ELSE default END",
    example: "SELECT \n  name,\n  salary,\n  CASE\n    WHEN salary >= 100000 THEN 'Senior'\n    WHEN salary >= 60000 THEN 'Mid-Level'\n    ELSE 'Junior'\n  END AS level\nFROM employees\nORDER BY salary DESC;",
  },
  "what is a stored procedure": {
    title: "Stored Procedures & Functions",
    content: "A stored procedure is a precompiled collection of SQL statements stored in the database. It can accept parameters, perform logic, and return results. Functions are similar but must return a value and can be used inside SQL expressions.\n\nBenefits:\n\n• Performance — precompiled execution plan\n• Security — controlled access to data\n• Reusability — call from multiple applications\n• Maintainability — change logic in one place\n\nProcedure vs Function:\n\n• Procedure — called with CALL/EXEC, can modify data, doesn't need to return\n• Function — called in expressions, should be side-effect free, must return a value",
    example: "CREATE PROCEDURE give_raise(\n  IN dept VARCHAR(50),\n  IN pct DECIMAL(5,2)\n)\nBEGIN\n  UPDATE employees\n  SET salary = salary * (1 + pct / 100)\n  WHERE department = dept;\nEND;\n\nCALL give_raise('Engineering', 10.0);",
  },
  "what is a trigger": {
    title: "Database Triggers",
    content: "A trigger is a special stored procedure that automatically executes when a specific event occurs on a table (INSERT, UPDATE, or DELETE). Triggers are useful for auditing, enforcing business rules, and maintaining data integrity.\n\nTiming:\n\n• BEFORE — runs before the event (can modify/cancel the operation)\n• AFTER — runs after the event (good for logging/auditing)\n• INSTEAD OF — replaces the event (used with views)\n\nScope:\n\n• FOR EACH ROW — fires once per affected row\n• FOR EACH STATEMENT — fires once per SQL statement",
    example: "CREATE TRIGGER audit_salary_change\nAFTER UPDATE OF salary ON employees\nFOR EACH ROW\nBEGIN\n  INSERT INTO salary_audit (employee_id, old_salary, new_salary, changed_at)\n  VALUES (OLD.id, OLD.salary, NEW.salary, NOW());\nEND;",
  },
};

const TOPIC_ALIASES: Record<string, string> = {
  "sql": "what is sql",
  "ddl": "ddl dml dcl tcl",
  "dml": "ddl dml dcl tcl",
  "dcl": "ddl dml dcl tcl",
  "tcl": "ddl dml dcl tcl",
  "ddl dml": "ddl dml dcl tcl",
  "ddl/dml": "ddl dml dcl tcl",
  "ddl/dml/dcl/tcl": "ddl dml dcl tcl",
  "ddl dml dcl": "ddl dml dcl tcl",
  "sql sub-languages": "ddl dml dcl tcl",
  "sql sublanguages": "ddl dml dcl tcl",
  "sql commands": "ddl dml dcl tcl",
  "types of sql": "ddl dml dcl tcl",
  "sql categories": "ddl dml dcl tcl",
  "joins": "what is a join",
  "join": "what is a join",
  "inner join": "what is a join",
  "left join": "what is a join",
  "right join": "what is a join",
  "outer join": "what is a join",
  "full outer join": "what is a join",
  "cross join": "what is a join",
  "self join": "what is a join",
  "where": "what is a where clause",
  "where clause": "what is a where clause",
  "group by": "what is group by",
  "groupby": "what is group by",
  "index": "what is an index",
  "indexes": "what is an index",
  "indexing": "what is an index",
  "primary key": "what is a primary key",
  "foreign key": "what is a primary key",
  "keys": "what is a primary key",
  "subquery": "what is a subquery",
  "subqueries": "what is a subquery",
  "nested query": "what is a subquery",
  "normalization": "what is normalization",
  "normal forms": "what is normalization",
  "1nf": "what is normalization",
  "2nf": "what is normalization",
  "3nf": "what is normalization",
  "bcnf": "what is normalization",
  "aggregate": "what are aggregate functions",
  "aggregates": "what are aggregate functions",
  "aggregate functions": "what are aggregate functions",
  "where vs having": "difference between where and having",
  "having": "difference between where and having",
  "having clause": "difference between where and having",
  "transaction": "what is a transaction",
  "transactions": "what is a transaction",
  "acid": "what is a transaction",
  "view": "what is a view",
  "views": "what is a view",
  "materialized view": "what is a view",
  "window function": "what are window functions",
  "window functions": "what are window functions",
  "row_number": "what are window functions",
  "rank": "what are window functions",
  "dense_rank": "what are window functions",
  "lag": "what are window functions",
  "lead": "what are window functions",
  "over": "what are window functions",
  "partition by": "what are window functions",
  "union": "what is union",
  "union all": "what is union",
  "intersect": "what is union",
  "except": "what is union",
  "set operations": "what is union",
  "order by": "what is order by",
  "sorting": "what is order by",
  "asc": "what is order by",
  "desc": "what is order by",
  "distinct": "what is distinct",
  "unique values": "what is distinct",
  "null": "what is null",
  "nulls": "what is null",
  "is null": "what is null",
  "coalesce": "what is null",
  "nullif": "what is null",
  "case": "what is case when",
  "case when": "what is case when",
  "if else": "what is case when",
  "conditional": "what is case when",
  "stored procedure": "what is a stored procedure",
  "stored procedures": "what is a stored procedure",
  "procedure": "what is a stored procedure",
  "function": "what is a stored procedure",
  "functions": "what is a stored procedure",
  "trigger": "what is a trigger",
  "triggers": "what is a trigger",
  "audit": "what is a trigger",
};

function findKnowledgeEntry(query: string) {
  const lower = query.toLowerCase().replace(/[?!.,]/g, "").trim();

  for (const [key, entry] of Object.entries(SQL_KNOWLEDGE)) {
    if (lower.includes(key) || lower === key) return entry;
  }

  for (const [alias, key] of Object.entries(TOPIC_ALIASES)) {
    if (lower.includes(alias) || lower === alias) return SQL_KNOWLEDGE[key];
  }

  const normalized = lower.replace(/\//g, " ").replace(/\s+/g, " ");
  for (const [alias, key] of Object.entries(TOPIC_ALIASES)) {
    if (normalized.includes(alias)) return SQL_KNOWLEDGE[key];
  }

  const words = normalized.split(/\s+/).filter(w => w.length > 2);
  for (const [alias, key] of Object.entries(TOPIC_ALIASES)) {
    if (words.some(w => w === alias)) return SQL_KNOWLEDGE[key];
  }

  return null;
}

function classifyIntent(query: string): "sql_generation" | "explanation" {
  const trimmed = query.trim();
  const lower = trimmed.toLowerCase();

  const explanationStarters = [
    "what is", "what are", "what's", "whats",
    "explain", "describe", "define", "tell me about",
    "how does", "how do", "how to", "why is", "why do", "why are",
    "difference between", "compare", "meaning of",
    "can you explain", "help me understand",
    "what does", "when to use", "when should",
    "types of",
  ];

  for (const starter of explanationStarters) {
    if (lower.startsWith(starter) || lower.includes(starter)) return "explanation";
  }

  if (findKnowledgeEntry(trimmed)) return "explanation";

  for (const pattern of SQL_GENERATION_PATTERNS) {
    if (pattern.test(trimmed)) return "sql_generation";
  }

  if (lower.endsWith("?")) return "explanation";

  const allTopicTerms = [
    ...Object.keys(SQL_KNOWLEDGE),
    ...Object.keys(TOPIC_ALIASES),
  ];
  const normalized = lower.replace(/[?!.,/]/g, " ").replace(/\s+/g, " ").trim();
  for (const term of allTopicTerms) {
    if (normalized === term || normalized.split(" ").every(w => term.includes(w))) {
      return "explanation";
    }
  }

  return "sql_generation";
}

function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

function formatTime(ts: number): string {
  const now = Date.now();
  const diff = now - ts;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "Just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days === 1) return "Yesterday";
  if (days < 7) return `${days}d ago`;
  return new Date(ts).toLocaleDateString();
}

function groupConversations(convs: Conversation[]) {
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const yesterday = today - 86400000;
  const weekAgo = today - 7 * 86400000;

  const groups: { label: string; items: Conversation[] }[] = [
    { label: "Today", items: [] },
    { label: "Yesterday", items: [] },
    { label: "Previous 7 days", items: [] },
    { label: "Older", items: [] },
  ];

  const sorted = [...convs].sort((a, b) => b.updatedAt - a.updatedAt);
  for (const c of sorted) {
    if (c.updatedAt >= today) groups[0].items.push(c);
    else if (c.updatedAt >= yesterday) groups[1].items.push(c);
    else if (c.updatedAt >= weekAgo) groups[2].items.push(c);
    else groups[3].items.push(c);
  }

  return groups.filter(g => g.items.length > 0);
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <button
      onClick={handleCopy}
      className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-medium text-claude-text-400 hover:text-claude-text-200 hover:bg-claude-bg-200 transition-colors"
    >
      {copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

function SqlBlock({ code, label }: { code: string; label?: string }) {
  return (
    <div className="mt-3 rounded-lg overflow-hidden border border-claude-bg-300/60">
      <div className="flex items-center justify-between px-3 py-1.5 bg-claude-bg-200/70">
        <span className="text-[11px] font-medium text-claude-text-400 uppercase tracking-wider">
          {label || "SQL"}
        </span>
        <CopyButton text={code} />
      </div>
      <pre className="px-3 py-3 bg-claude-bg-0 overflow-x-auto custom-scrollbar">
        <code className="text-xs font-mono text-claude-text-200 leading-relaxed">{code}</code>
      </pre>
    </div>
  );
}

function ExplanationContent({ content }: { content: string }) {
  const parts = content.split("\n\n");
  return (
    <div className="space-y-2.5">
      {parts.map((block, i) => {
        if (block.startsWith("•") || block.includes("\n•")) {
          const lines = block.split("\n").filter(Boolean);
          return (
            <ul key={i} className="space-y-1">
              {lines.map((line, j) => (
                <li key={j} className="text-sm text-claude-text-200 leading-relaxed flex gap-2">
                  {line.startsWith("•") ? (
                    <>
                      <span className="text-claude-accent mt-0.5 flex-shrink-0">•</span>
                      <span>{line.slice(1).trim()}</span>
                    </>
                  ) : (
                    <span>{line}</span>
                  )}
                </li>
              ))}
            </ul>
          );
        }
        return (
          <p key={i} className="text-sm text-claude-text-200 leading-relaxed">
            {block}
          </p>
        );
      })}
    </div>
  );
}

function SchemaPanel({
  schema,
  onChange,
}: {
  schema: string;
  onChange: (s: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="border-b border-claude-bg-300/50">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 w-full px-4 py-2.5 text-xs font-medium text-claude-text-300 hover:text-claude-text-200 hover:bg-claude-bg-200/50 transition-colors"
      >
        <Database className="w-3.5 h-3.5" />
        <span>Schema Context</span>
        <ChevronRight className={cn("w-3 h-3 ml-auto transition-transform", expanded && "rotate-90")} />
      </button>
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <textarea
              value={schema}
              onChange={(e) => onChange(e.target.value)}
              className="w-full h-40 px-4 py-2 text-xs font-mono bg-claude-bg-100 text-claude-text-200 resize-none outline-none border-t border-claude-bg-300/50 custom-scrollbar"
              placeholder="Paste your CREATE TABLE statements here..."
              spellCheck={false}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function MessageBubble({ msg }: { msg: ChatMessage }) {
  if (msg.role === "user") {
    return (
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.25 }}
        className="flex gap-3 justify-end"
      >
        <div className="max-w-[80%] rounded-2xl rounded-br-md bg-claude-accent text-white px-4 py-3">
          <p className="text-sm leading-relaxed">{msg.content}</p>
          <p className="text-[10px] text-white/50 text-right mt-2">{formatTime(msg.timestamp)}</p>
        </div>
        <div className="flex-shrink-0 w-7 h-7 rounded-full bg-claude-accent flex items-center justify-center text-white text-xs font-semibold mt-0.5">
          R
        </div>
      </motion.div>
    );
  }

  const isError = msg.type === "error";
  const isExplanation = msg.type === "explanation";

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className="flex gap-3 justify-start"
    >
      <div className={cn(
        "flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center mt-0.5",
        isExplanation ? "bg-blue-100" : "bg-claude-accent/10"
      )}>
        {isExplanation
          ? <BookOpen className="w-3.5 h-3.5 text-blue-600" />
          : <Sparkles className="w-3.5 h-3.5 text-claude-accent" />
        }
      </div>

      <div className={cn(
        "max-w-[85%] rounded-2xl rounded-bl-md px-4 py-3",
        isError
          ? "bg-red-50 border border-red-200"
          : "bg-claude-bg-100 border border-claude-bg-300/60"
      )}>
        {isError && (
          <div className="flex items-center gap-1.5 mb-1.5 text-red-600">
            <AlertCircle className="w-3.5 h-3.5" />
            <span className="text-xs font-medium">Error</span>
          </div>
        )}

        {isExplanation && msg.content && (
          <div className="flex items-center gap-1.5 mb-2 pb-2 border-b border-claude-bg-300/40">
            <span className="text-xs font-semibold text-claude-text-100">{msg.content}</span>
          </div>
        )}

        {isError ? (
          <p className="text-sm leading-relaxed text-red-600">{msg.content}</p>
        ) : isExplanation && msg.exampleSql ? (
          <>
            <ExplanationContent content={msg.sql || ""} />
            <SqlBlock code={msg.exampleSql} label="Example" />
          </>
        ) : (
          <>
            {!isExplanation && (
              <p className="text-sm leading-relaxed text-claude-text-200">{msg.content}</p>
            )}
            {msg.sql && !isExplanation && <SqlBlock code={msg.sql} />}
          </>
        )}

        {!isError && !isExplanation && (msg.confidence != null || msg.latency != null) && (
          <div className="flex items-center gap-3 mt-2.5 pt-2 border-t border-claude-bg-300/40">
            {msg.confidence != null && (
              <span className={cn(
                "text-[11px] font-medium px-2 py-0.5 rounded-full",
                msg.confidence >= 0.8 ? "bg-emerald-50 text-emerald-600" :
                msg.confidence >= 0.5 ? "bg-amber-50 text-amber-600" :
                "bg-red-50 text-red-500"
              )}>
                {(msg.confidence * 100).toFixed(0)}% confidence
              </span>
            )}
            {msg.latency != null && (
              <span className="text-[11px] text-claude-text-400 flex items-center gap-1">
                <Clock className="w-3 h-3" />
                {msg.latency.toFixed(0)}ms
              </span>
            )}
          </div>
        )}

        <p className={cn(
          "text-[10px] mt-2",
          isError ? "text-red-400" : "text-claude-text-500"
        )}>
          {formatTime(msg.timestamp)}
        </p>
      </div>
    </motion.div>
  );
}

export default function ChatPage() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [loading, setLoading] = useState(false);
  const [loadingType, setLoadingType] = useState<"sql" | "explanation">("sql");
  const [connected, setConnected] = useState<boolean | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const activeConv = conversations.find(c => c.id === activeId) || null;

  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) setConversations(JSON.parse(stored) as Conversation[]);
      const storedActive = localStorage.getItem(ACTIVE_KEY);
      if (storedActive) setActiveId(storedActive);
    } catch {}
  }, []);

  useEffect(() => {
    if (conversations.length > 0) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
    }
  }, [conversations]);

  useEffect(() => {
    if (activeId) localStorage.setItem(ACTIVE_KEY, activeId);
  }, [activeId]);

  useEffect(() => {
    const check = async () => setConnected(await checkHealth());
    check();
    const interval = setInterval(check, 15000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeConv?.messages.length, loading]);

  const updateConversation = useCallback((id: string, updater: (c: Conversation) => Conversation) => {
    setConversations(prev => prev.map(c => c.id === id ? updater(c) : c));
  }, []);

  const createNewChat = useCallback(() => {
    const newConv: Conversation = {
      id: generateId(),
      title: "New conversation",
      messages: [],
      schema: DEFAULT_SCHEMA,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    setConversations(prev => [newConv, ...prev]);
    setActiveId(newConv.id);
  }, []);

  const deleteConversation = useCallback((id: string) => {
    setConversations(prev => prev.filter(c => c.id !== id));
    if (activeId === id) setActiveId(null);
  }, [activeId]);

  const handleSend = useCallback(async (data: {
    message: string;
    files: AttachedFile[];
    model: string;
    isThinkingEnabled: boolean;
  }) => {
    if (!data.message.trim()) return;

    let convId = activeId;
    let schema = DEFAULT_SCHEMA;

    if (!convId) {
      const newConv: Conversation = {
        id: generateId(),
        title: data.message.slice(0, 60),
        messages: [],
        schema: DEFAULT_SCHEMA,
        createdAt: Date.now(),
        updatedAt: Date.now(),
      };
      setConversations(prev => [newConv, ...prev]);
      setActiveId(newConv.id);
      convId = newConv.id;
      schema = newConv.schema;
    } else {
      const conv = conversations.find(c => c.id === convId);
      schema = conv?.schema || DEFAULT_SCHEMA;
    }

    const userMsg: ChatMessage = {
      id: generateId(),
      role: "user",
      content: data.message,
      timestamp: Date.now(),
    };

    const currentConvId = convId;

    updateConversation(currentConvId, c => {
      const title = c.messages.length === 0 ? data.message.slice(0, 60) : c.title;
      return { ...c, title, messages: [...c.messages, userMsg], updatedAt: Date.now() };
    });

    const intent = classifyIntent(data.message);

    if (intent === "explanation") {
      setLoading(true);
      setLoadingType("explanation");

      await new Promise(r => setTimeout(r, 400 + Math.random() * 400));

      const entry = findKnowledgeEntry(data.message);

      const assistantMsg: ChatMessage = entry
        ? {
            id: generateId(),
            role: "assistant",
            type: "explanation",
            content: entry.title,
            sql: entry.content,
            exampleSql: entry.example,
            timestamp: Date.now(),
          }
        : {
            id: generateId(),
            role: "assistant",
            type: "explanation",
            content: "SQL Concept",
            sql: `I don't have a detailed entry for that specific topic yet, but here's what I can help with:\n\nYou can ask me to explain SQL concepts like:\n\n• "What is SQL?"\n• "Explain JOINs"\n• "What is GROUP BY?"\n• "Difference between WHERE and HAVING"\n• "What is a primary key?"\n• "Explain subqueries"\n• "What are aggregate functions?"\n• "What is normalization?"\n• "What is a transaction?"\n\nOr ask me to generate SQL queries like:\n\n• "Show all employees with salary above 50000"\n• "Find departments with more than 10 employees"\n• "Get the average salary by department"`,
            timestamp: Date.now(),
          };

      updateConversation(currentConvId, c => ({
        ...c,
        messages: [...c.messages, assistantMsg],
        updatedAt: Date.now(),
      }));

      setLoading(false);
      return;
    }

    setLoading(true);
    setLoadingType("sql");

    try {
      const response = await generateSQL({ question: data.message, schema });

      const assistantMsg: ChatMessage = {
        id: generateId(),
        role: "assistant",
        type: "sql",
        content: response.generated_sql
          ? "Here's the generated SQL query for your question:"
          : "I couldn't generate a SQL query for that question. Try rephrasing or check the schema context.",
        sql: response.generated_sql,
        confidence: response.confidence,
        latency: response.latency_ms,
        timestamp: Date.now(),
      };

      updateConversation(currentConvId, c => ({
        ...c,
        messages: [...c.messages, assistantMsg],
        updatedAt: Date.now(),
      }));
    } catch (err: unknown) {
      const errorMessage = (err && typeof err === "object" && "message" in err)
        ? (err as { message: string }).message
        : "Failed to generate SQL. Please check if the API server is running.";

      const errorMsg: ChatMessage = {
        id: generateId(),
        role: "assistant",
        type: "error",
        content: errorMessage,
        timestamp: Date.now(),
      };

      updateConversation(currentConvId, c => ({
        ...c,
        messages: [...c.messages, errorMsg],
        updatedAt: Date.now(),
      }));
    } finally {
      setLoading(false);
    }
  }, [activeId, conversations, updateConversation]);

  const currentHour = new Date().getHours();
  let greeting = "Good morning";
  if (currentHour >= 12 && currentHour < 18) greeting = "Good afternoon";
  else if (currentHour >= 18) greeting = "Good evening";

  const groups = groupConversations(conversations);
  const showHero = !activeConv || activeConv.messages.length === 0;

  return (
    <div className="flex h-[calc(100vh-49px)] bg-claude-bg-0 font-sans text-claude-text-100">
      <AnimatePresence>
        {sidebarOpen && (
          <motion.aside
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 280, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="flex-shrink-0 border-r border-claude-bg-300/50 bg-claude-bg-100 flex flex-col overflow-hidden"
          >
            <div className="flex items-center justify-between px-3 py-3 border-b border-claude-bg-300/50">
              <button
                onClick={createNewChat}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium text-claude-text-200 hover:bg-claude-bg-200 transition-colors"
              >
                <Plus className="w-4 h-4" />
                New chat
              </button>
              <button
                onClick={() => setSidebarOpen(false)}
                className="p-1.5 rounded-lg text-claude-text-400 hover:text-claude-text-200 hover:bg-claude-bg-200 transition-colors"
              >
                <PanelLeftClose className="w-4 h-4" />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto custom-scrollbar px-2 py-2">
              {groups.length === 0 && (
                <div className="flex flex-col items-center justify-center h-full text-center p-4 opacity-50">
                  <MessageSquare className="w-8 h-8 text-claude-text-400 mb-2" />
                  <p className="text-xs text-claude-text-400">No conversations yet</p>
                </div>
              )}
              {groups.map(group => (
                <div key={group.label} className="mb-3">
                  <p className="px-2 py-1.5 text-[11px] font-medium text-claude-text-400 uppercase tracking-wider">
                    {group.label}
                  </p>
                  {group.items.map(conv => (
                    <div
                      key={conv.id}
                      onClick={() => setActiveId(conv.id)}
                      className={cn(
                        "group flex items-center gap-2 px-2.5 py-2 rounded-lg cursor-pointer transition-colors mb-0.5",
                        activeId === conv.id
                          ? "bg-claude-bg-200 text-claude-text-100"
                          : "text-claude-text-300 hover:bg-claude-bg-200/50 hover:text-claude-text-200"
                      )}
                    >
                      <MessageSquare className="w-3.5 h-3.5 flex-shrink-0" />
                      <span className="text-sm truncate flex-1">{conv.title}</span>
                      <button
                        onClick={(e) => { e.stopPropagation(); deleteConversation(conv.id); }}
                        className="p-1 rounded opacity-0 group-hover:opacity-100 hover:bg-claude-bg-300/50 hover:text-red-500 transition-all"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  ))}
                </div>
              ))}
            </div>

            <div className="px-3 py-2.5 border-t border-claude-bg-300/50">
              <div className={cn(
                "flex items-center gap-2 text-[11px] font-medium px-2 py-1 rounded-md",
                connected === true && "text-emerald-600",
                connected === false && "text-red-500",
                connected === null && "text-claude-text-400"
              )}>
                {connected === true ? <Wifi className="w-3 h-3" /> : connected === false ? <WifiOff className="w-3 h-3" /> : <Clock className="w-3 h-3 animate-pulse" />}
                {connected === true ? "API Connected" : connected === false ? "API Disconnected" : "Checking..."}
              </div>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      <div className="flex-1 flex flex-col min-w-0">
        {!sidebarOpen && (
          <div className="flex items-center gap-2 px-3 py-2 border-b border-claude-bg-300/30">
            <button
              onClick={() => setSidebarOpen(true)}
              className="p-1.5 rounded-lg text-claude-text-400 hover:text-claude-text-200 hover:bg-claude-bg-200 transition-colors"
            >
              <PanelLeft className="w-4 h-4" />
            </button>
            <button
              onClick={createNewChat}
              className="p-1.5 rounded-lg text-claude-text-400 hover:text-claude-text-200 hover:bg-claude-bg-200 transition-colors"
            >
              <Plus className="w-4 h-4" />
            </button>
          </div>
        )}

        {activeConv && (
          <SchemaPanel
            schema={activeConv.schema}
            onChange={(s) => updateConversation(activeConv.id, c => ({ ...c, schema: s }))}
          />
        )}

        <div className="flex-1 overflow-y-auto custom-scrollbar">
          {showHero ? (
            <div className="flex flex-col items-center justify-center h-full p-4 animate-claude-fade-in">
              <div className="w-20 h-20 mx-auto mb-6 flex items-center justify-center">
                <svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" className="w-full h-full">
                  <defs>
                    <ellipse id="hero-petal" cx="100" cy="100" rx="90" ry="22" />
                  </defs>
                  <g fill="#D46B4F" fillRule="evenodd">
                    <use href="#hero-petal" transform="rotate(0 100 100)" />
                    <use href="#hero-petal" transform="rotate(45 100 100)" />
                    <use href="#hero-petal" transform="rotate(90 100 100)" />
                    <use href="#hero-petal" transform="rotate(135 100 100)" />
                  </g>
                </svg>
              </div>
              <h1 className="text-3xl sm:text-4xl font-serif font-light text-claude-text-200 mb-3 tracking-tight text-center">
                {greeting},{" "}
                <span className="relative inline-block pb-2">
                  Rajat
                  <svg
                    className="absolute w-[140%] h-[20px] -bottom-1 -left-[5%] text-claude-accent"
                    viewBox="0 0 140 24"
                    fill="none"
                    preserveAspectRatio="none"
                    aria-hidden="true"
                  >
                    <path d="M6 16 Q 70 24, 134 14" stroke="currentColor" strokeWidth="3" strokeLinecap="round" fill="none" />
                  </svg>
                </span>
              </h1>
              <p className="text-sm text-claude-text-400 mb-8 text-center max-w-md">
                Generate SQL queries from natural language or ask about SQL concepts.
              </p>

              <div className="w-full max-w-xl space-y-4">
                <div>
                  <div className="flex items-center gap-2 mb-2 justify-center">
                    <Zap className="w-3.5 h-3.5 text-claude-accent" />
                    <span className="text-xs font-medium text-claude-text-300 uppercase tracking-wider">Generate SQL</span>
                  </div>
                  <div className="flex flex-wrap justify-center gap-2">
                    {[
                      "Show all employees with salary above 50000",
                      "Get the average salary by department",
                      "Find departments with more than 10 employees",
                    ].map((q) => (
                      <button
                        key={q}
                        onClick={() => handleSend({ message: q, files: [], model: MODELS[0].id, isThinkingEnabled: false })}
                        className="px-3 py-1.5 text-xs text-claude-text-300 border border-claude-bg-300 rounded-full hover:bg-claude-bg-200 hover:text-claude-text-200 transition-colors"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <div className="flex items-center gap-2 mb-2 justify-center">
                    <BookOpen className="w-3.5 h-3.5 text-blue-500" />
                    <span className="text-xs font-medium text-claude-text-300 uppercase tracking-wider">Learn SQL</span>
                  </div>
                  <div className="flex flex-wrap justify-center gap-2">
                    {[
                      "What is SQL?",
                      "Explain JOINs",
                      "What is GROUP BY?",
                      "Difference between WHERE and HAVING",
                    ].map((q) => (
                      <button
                        key={q}
                        onClick={() => handleSend({ message: q, files: [], model: MODELS[0].id, isThinkingEnabled: false })}
                        className="px-3 py-1.5 text-xs text-claude-text-300 border border-blue-200 rounded-full hover:bg-blue-50 hover:text-blue-600 transition-colors"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto px-4 py-6 space-y-6">
              {activeConv?.messages.map((msg) => (
                <MessageBubble key={msg.id} msg={msg} />
              ))}

              {loading && (
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex gap-3"
                >
                  <div className={cn(
                    "flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center",
                    loadingType === "explanation" ? "bg-blue-100" : "bg-claude-accent/10"
                  )}>
                    {loadingType === "explanation"
                      ? <BookOpen className="w-3.5 h-3.5 text-blue-600" />
                      : <Sparkles className="w-3.5 h-3.5 text-claude-accent" />
                    }
                  </div>
                  <div className="bg-claude-bg-100 border border-claude-bg-300/60 rounded-2xl rounded-bl-md px-4 py-3">
                    <div className="flex items-center gap-2">
                      <div className="flex gap-1">
                        <span className={cn("w-1.5 h-1.5 rounded-full animate-pulse", loadingType === "explanation" ? "bg-blue-400" : "bg-claude-accent/60")} style={{ animationDelay: "0ms" }} />
                        <span className={cn("w-1.5 h-1.5 rounded-full animate-pulse", loadingType === "explanation" ? "bg-blue-400" : "bg-claude-accent/60")} style={{ animationDelay: "200ms" }} />
                        <span className={cn("w-1.5 h-1.5 rounded-full animate-pulse", loadingType === "explanation" ? "bg-blue-400" : "bg-claude-accent/60")} style={{ animationDelay: "400ms" }} />
                      </div>
                      <span className="text-xs text-claude-text-400">
                        {loadingType === "explanation" ? "Looking up concept..." : "Generating SQL..."}
                      </span>
                    </div>
                  </div>
                </motion.div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <div className="flex-shrink-0 px-4 pb-4 pt-2">
          <ClaudeChatInput
            onSendMessage={handleSend}
            customModels={MODELS}
            defaultModel="lora-finetuned"
            loading={loading}
          />
        </div>
      </div>
    </div>
  );
}