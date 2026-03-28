export interface KnowledgeEntry {
  title: string;
  content: string;
  example: string;
  tags: string[];
}

export const SQL_KNOWLEDGE: KnowledgeEntry[] = [
  {
    title: "What is SQL?",
    content: "SQL (Structured Query Language) is a standard programming language designed for managing and manipulating relational databases. It allows you to create, read, update, and delete data stored in tables.\n\nSQL is divided into sub-languages:\n\n• DDL (Data Definition Language) — CREATE, ALTER, DROP, TRUNCATE\n• DML (Data Manipulation Language) — SELECT, INSERT, UPDATE, DELETE\n• DCL (Data Control Language) — GRANT, REVOKE\n• TCL (Transaction Control Language) — COMMIT, ROLLBACK, SAVEPOINT",
    example: "SELECT name, salary\nFROM employees\nWHERE department = 'Engineering'\nORDER BY salary DESC;",
    tags: ["sql", "structured query language", "database language", "rdbms", "relational database"],
  },
  {
    title: "DDL / DML / DCL / TCL — SQL Sub-Languages",
    content: "SQL commands are grouped into four categories:\n\nDDL (Data Definition Language) defines database structure:\n\n• CREATE — create tables, views, indexes\n• ALTER — modify existing structures\n• DROP — permanently remove objects\n• TRUNCATE — remove all rows from a table\n• RENAME — rename an object\n\nDML (Data Manipulation Language) works with data:\n\n• SELECT — retrieve/query data\n• INSERT — add new rows\n• UPDATE — modify existing rows\n• DELETE — remove specific rows\n\nDCL (Data Control Language) manages permissions:\n\n• GRANT — give privileges to users/roles\n• REVOKE — remove privileges\n\nTCL (Transaction Control Language) manages transactions:\n\n• COMMIT — save changes permanently\n• ROLLBACK — undo changes since last commit\n• SAVEPOINT — create a restore point",
    example: "-- DDL\nCREATE TABLE products (id INT PRIMARY KEY, name VARCHAR(100), price DECIMAL(10,2));\n\n-- DML\nINSERT INTO products VALUES (1, 'Widget', 29.99);\nSELECT * FROM products WHERE price > 20;\n\n-- DCL\nGRANT SELECT, INSERT ON products TO analyst_role;\n\n-- TCL\nBEGIN;\nUPDATE products SET price = 24.99 WHERE id = 1;\nCOMMIT;",
    tags: ["ddl", "dml", "dcl", "tcl", "data definition language", "data manipulation language", "data control language", "transaction control language", "sql sub-languages", "sql sublanguages", "types of sql", "sql categories", "sql commands", "ddl/dml/dcl/tcl", "ddl dml dcl tcl", "ddl dml"],
  },
  {
    title: "SQL JOINs",
    content: "A JOIN clause combines rows from two or more tables based on a related column. JOINs are fundamental to relational databases.\n\nTypes of JOINs:\n\n• INNER JOIN — returns only matching rows from both tables\n• LEFT JOIN (LEFT OUTER JOIN) — all rows from left table, matches from right\n• RIGHT JOIN (RIGHT OUTER JOIN) — all rows from right table, matches from left\n• FULL OUTER JOIN — all rows when there's a match in either table\n• CROSS JOIN — Cartesian product of both tables\n• SELF JOIN — a table joined with itself\n• NATURAL JOIN — joins on columns with same name automatically",
    example: "SELECT e.name, d.name AS department\nFROM employees e\nINNER JOIN departments d ON e.department_id = d.id;\n\nSELECT e.name, m.name AS manager\nFROM employees e\nLEFT JOIN employees m ON e.manager_id = m.id;",
    tags: ["join", "joins", "inner join", "left join", "right join", "outer join", "full outer join", "cross join", "self join", "natural join", "table join", "combining tables"],
  },
  {
    title: "WHERE Clause",
    content: "The WHERE clause filters rows based on conditions. It's used with SELECT, UPDATE, and DELETE to target specific records.\n\nCommon operators:\n\n• = , != , <> — equality / inequality\n• < , > , <= , >= — comparison\n• BETWEEN — range check\n• LIKE — pattern matching (% any chars, _ single char)\n• IN — match against a list\n• IS NULL / IS NOT NULL — null checks\n• EXISTS — check if subquery returns rows\n• AND, OR, NOT — combine conditions",
    example: "SELECT name, salary\nFROM employees\nWHERE department = 'Sales'\n  AND salary BETWEEN 40000 AND 80000\n  AND name LIKE 'J%'\n  AND hire_date IS NOT NULL;",
    tags: ["where", "where clause", "filter", "filtering", "condition", "conditions"],
  },
  {
    title: "GROUP BY Clause",
    content: "GROUP BY groups rows sharing values in specified columns into summary rows. It's used with aggregate functions (COUNT, SUM, AVG, MAX, MIN) to compute statistics per group.\n\nHAVING filters groups after aggregation (WHERE filters rows before grouping).\n\nExecution order: FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY",
    example: "SELECT department, COUNT(*) AS total, AVG(salary) AS avg_salary\nFROM employees\nGROUP BY department\nHAVING COUNT(*) > 5\nORDER BY avg_salary DESC;",
    tags: ["group by", "groupby", "grouping", "aggregation", "group"],
  },
  {
    title: "Database Indexes",
    content: "An index is a data structure that speeds up data retrieval at the cost of additional storage and slower writes. Like a book's index, it lets the database jump directly to matching rows.\n\nTypes:\n\n• B-Tree index — default, good for equality and range queries\n• Hash index — fast for exact equality only\n• Composite index — multiple columns\n• Unique index — enforces uniqueness\n• Partial index — indexes only rows matching a condition\n• Full-text index — for text search\n\nWhen to use:\n\n• Columns in WHERE, JOIN, ORDER BY clauses\n• High-cardinality columns\n\nWhen to avoid:\n\n• Small tables, columns with many NULLs, heavy-write tables",
    example: "CREATE INDEX idx_emp_dept ON employees(department);\nCREATE UNIQUE INDEX idx_emp_email ON employees(email);\nCREATE INDEX idx_composite ON employees(department, salary);",
    tags: ["index", "indexes", "indexing", "b-tree", "hash index", "composite index", "unique index", "database index", "performance", "query optimization"],
  },
  {
    title: "Primary Keys & Foreign Keys",
    content: "A PRIMARY KEY uniquely identifies each record. It must be unique and non-null. Each table has one primary key (can span multiple columns as a composite key).\n\nA FOREIGN KEY references the primary key of another table, creating relationships and enforcing referential integrity.\n\nOther key types:\n\n• Candidate key — any column(s) that could be a primary key\n• Composite key — primary key using multiple columns\n• Surrogate key — auto-generated ID (no business meaning)\n• Natural key — has business meaning (email, SSN)",
    example: "CREATE TABLE departments (\n  id INT PRIMARY KEY,\n  name VARCHAR(50) UNIQUE NOT NULL\n);\n\nCREATE TABLE employees (\n  id INT PRIMARY KEY,\n  name VARCHAR(100),\n  dept_id INT,\n  FOREIGN KEY (dept_id) REFERENCES departments(id)\n    ON DELETE SET NULL ON UPDATE CASCADE\n);",
    tags: ["primary key", "foreign key", "keys", "pk", "fk", "candidate key", "composite key", "surrogate key", "natural key", "referential integrity"],
  },
  {
    title: "Subqueries",
    content: "A subquery is a query nested inside another query. It can appear in SELECT, FROM, WHERE, or HAVING clauses.\n\nTypes:\n\n• Scalar subquery — returns a single value\n• Row subquery — returns a single row\n• Table subquery — returns a result set (used in FROM)\n• Correlated subquery — references the outer query (runs once per outer row)\n\nSubquery vs JOIN: JOINs are usually faster for combining data. Subqueries are clearer for filtering based on aggregates or existence checks.",
    example: "SELECT name, salary\nFROM employees\nWHERE salary > (SELECT AVG(salary) FROM employees);\n\nSELECT department, (SELECT COUNT(*) FROM employees e WHERE e.dept_id = d.id) AS headcount\nFROM departments d;",
    tags: ["subquery", "subqueries", "nested query", "nested queries", "inner query", "correlated subquery", "scalar subquery"],
  },
  {
    title: "Database Normalization",
    content: "Normalization organizes tables to minimize redundancy and dependency by dividing large tables into smaller ones linked by relationships.\n\nNormal Forms:\n\n• 1NF — each cell holds a single value, no repeating groups\n• 2NF — 1NF + all non-key columns depend on the entire primary key\n• 3NF — 2NF + no transitive dependencies (non-key depends only on PK)\n• BCNF — every determinant is a candidate key\n• 4NF — no multi-valued dependencies\n• 5NF — no join dependencies\n\nDenormalization is the reverse — adding redundancy for read performance (common in data warehouses).",
    example: "-- Denormalized (bad):\n-- orders(id, product, customer_name, customer_email)\n\n-- Normalized (3NF):\nCREATE TABLE customers (id INT PRIMARY KEY, name VARCHAR(100), email VARCHAR(100));\nCREATE TABLE orders (id INT PRIMARY KEY, product VARCHAR(100), customer_id INT REFERENCES customers(id));",
    tags: ["normalization", "normal forms", "1nf", "2nf", "3nf", "bcnf", "4nf", "5nf", "denormalization", "database design", "redundancy"],
  },
  {
    title: "Aggregate Functions",
    content: "Aggregate functions compute a single result from a set of values. Used with GROUP BY for per-group statistics.\n\nCommon aggregates:\n\n• COUNT(*) — total rows (including NULLs)\n• COUNT(col) — non-NULL values in column\n• SUM(col) — total of numeric column\n• AVG(col) — average value\n• MAX(col) — highest value\n• MIN(col) — lowest value\n• GROUP_CONCAT() / STRING_AGG() — concatenate values\n• ARRAY_AGG() — collect into array (PostgreSQL)",
    example: "SELECT department,\n  COUNT(*) AS headcount,\n  SUM(salary) AS total_payroll,\n  ROUND(AVG(salary), 2) AS avg_salary,\n  MAX(salary) AS top_salary,\n  MIN(salary) AS entry_salary\nFROM employees\nGROUP BY department;",
    tags: ["aggregate", "aggregates", "aggregate functions", "count", "sum", "avg", "max", "min", "group_concat", "string_agg"],
  },
  {
    title: "WHERE vs HAVING",
    content: "Both filter data but at different stages:\n\nWHERE filters individual rows before grouping. Cannot use aggregate functions.\n\nHAVING filters groups after GROUP BY. Can use aggregate functions.\n\nExecution order: FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY\n\nRule of thumb: use WHERE to filter rows, HAVING to filter groups.",
    example: "SELECT department, AVG(salary) AS avg_sal\nFROM employees\nWHERE hire_date > '2023-01-01'     -- filters rows first\nGROUP BY department\nHAVING AVG(salary) > 60000;        -- then filters groups",
    tags: ["where vs having", "having", "having clause", "where having difference", "having vs where"],
  },
  {
    title: "Database Transactions & ACID",
    content: "A transaction is a sequence of operations performed as a single logical unit. Transactions follow ACID properties:\n\n• Atomicity — all succeed or all fail (no partial commits)\n• Consistency — database stays in a valid state\n• Isolation — concurrent transactions don't interfere\n• Durability — committed changes survive crashes\n\nIsolation levels (from least to most strict):\n\n• READ UNCOMMITTED — can see uncommitted changes (dirty reads)\n• READ COMMITTED — only sees committed data\n• REPEATABLE READ — same query returns same results within transaction\n• SERIALIZABLE — full isolation, as if transactions ran sequentially",
    example: "BEGIN TRANSACTION;\n\nUPDATE accounts SET balance = balance - 500 WHERE id = 1;\nUPDATE accounts SET balance = balance + 500 WHERE id = 2;\n\n-- If everything is OK:\nCOMMIT;\n\n-- If something went wrong:\n-- ROLLBACK;",
    tags: ["transaction", "transactions", "acid", "atomicity", "consistency", "isolation", "durability", "commit", "rollback", "savepoint", "isolation level", "read committed", "serializable"],
  },
  {
    title: "SQL Views",
    content: "A view is a virtual table based on a SELECT query. It doesn't store data — it pulls from underlying tables each time.\n\nTypes:\n\n• Regular view — a saved query, always up to date\n• Materialized view — stores results physically (needs refresh)\n• Updatable view — allows INSERT/UPDATE/DELETE on simple views\n\nUse cases:\n\n• Simplify complex queries\n• Security — expose only certain columns/rows\n• Abstraction — hide table structure changes",
    example: "CREATE VIEW high_earners AS\nSELECT name, department, salary\nFROM employees WHERE salary > 80000;\n\nSELECT * FROM high_earners ORDER BY salary DESC;\n\n-- Materialized (PostgreSQL):\nCREATE MATERIALIZED VIEW dept_stats AS\nSELECT department, AVG(salary) as avg_sal FROM employees GROUP BY department;",
    tags: ["view", "views", "materialized view", "virtual table", "create view"],
  },
  {
    title: "Window Functions",
    content: "Window functions perform calculations across a set of related rows without collapsing them (unlike GROUP BY). They use OVER() to define the window.\n\nCommon window functions:\n\n• ROW_NUMBER() — sequential number within partition\n• RANK() — rank with gaps for ties\n• DENSE_RANK() — rank without gaps\n• NTILE(n) — divide rows into n groups\n• LAG(col, n) — value from n rows before\n• LEAD(col, n) — value from n rows after\n• FIRST_VALUE() / LAST_VALUE() — first/last in window\n• SUM/AVG/COUNT OVER() — running totals/averages\n\nFrame clauses: ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
    example: "SELECT name, department, salary,\n  RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS dept_rank,\n  salary - LAG(salary) OVER (ORDER BY salary) AS diff_from_prev,\n  SUM(salary) OVER (ORDER BY hire_date) AS running_total\nFROM employees;",
    tags: ["window function", "window functions", "over", "partition by", "row_number", "rank", "dense_rank", "ntile", "lag", "lead", "first_value", "last_value", "running total", "analytical functions", "analytic functions"],
  },
  {
    title: "UNION & Set Operations",
    content: "Set operations combine results from multiple SELECT statements. All queries must have the same number of columns with compatible types.\n\n• UNION — combines results, removes duplicates\n• UNION ALL — combines results, keeps duplicates (faster)\n• INTERSECT — only rows present in both results\n• EXCEPT / MINUS — rows in first result not in second",
    example: "SELECT name, 'Employee' AS role FROM employees\nUNION ALL\nSELECT name, 'Manager' AS role FROM employees WHERE id IN (\n  SELECT manager_id FROM departments\n);",
    tags: ["union", "union all", "intersect", "except", "minus", "set operations", "combine queries", "combine results"],
  },
  {
    title: "ORDER BY & Sorting",
    content: "ORDER BY sorts the result set. Default is ascending (ASC). Use DESC for descending.\n\nFeatures:\n\n• Sort by multiple columns: ORDER BY col1 ASC, col2 DESC\n• Sort by column position: ORDER BY 1, 2\n• Sort by expression: ORDER BY salary * 12\n• Sort by alias: ORDER BY annual_salary\n• NULL handling: NULLS FIRST / NULLS LAST (PostgreSQL)\n\nLIMIT/OFFSET for pagination:\n\n• LIMIT n — return only n rows\n• OFFSET m — skip first m rows\n• SQL Server uses TOP n instead of LIMIT",
    example: "SELECT name, department, salary\nFROM employees\nORDER BY department ASC, salary DESC\nLIMIT 20 OFFSET 0;\n\n-- Pagination (page 3, 10 per page):\nSELECT * FROM employees ORDER BY id LIMIT 10 OFFSET 20;",
    tags: ["order by", "sorting", "sort", "asc", "desc", "ascending", "descending", "limit", "offset", "pagination", "top"],
  },
  {
    title: "DISTINCT Keyword",
    content: "DISTINCT eliminates duplicate rows from the result set.\n\nVariations:\n\n• SELECT DISTINCT — unique rows across all selected columns\n• COUNT(DISTINCT col) — count unique values\n• DISTINCT ON (col) — first row per unique value (PostgreSQL)\n\nAlternative approaches to deduplication:\n\n• GROUP BY — when you also need aggregation\n• ROW_NUMBER() OVER (PARTITION BY ...) — more control over which duplicate to keep\n• EXISTS — for checking existence without duplicates",
    example: "SELECT DISTINCT department FROM employees;\n\nSELECT COUNT(DISTINCT department) AS unique_depts FROM employees;\n\n-- Keep most recent per employee:\nSELECT * FROM (\n  SELECT *, ROW_NUMBER() OVER (PARTITION BY employee_id ORDER BY created_at DESC) AS rn\n  FROM logs\n) t WHERE rn = 1;",
    tags: ["distinct", "unique", "unique values", "deduplication", "remove duplicates", "duplicates"],
  },
  {
    title: "NULL Handling in SQL",
    content: "NULL represents missing/unknown data. It's not zero, not empty string, not false.\n\nKey behaviors:\n\n• NULL = NULL → NULL (not true!) — use IS NULL\n• NULL + 5 → NULL — any arithmetic with NULL returns NULL\n• COUNT(*) counts NULLs, COUNT(col) does not\n• NULLs are excluded from aggregate functions (AVG, SUM)\n\nFunctions for NULLs:\n\n• COALESCE(a, b, c) — returns first non-NULL value\n• NULLIF(a, b) — returns NULL if a = b\n• IFNULL(a, b) — MySQL: returns b if a is NULL\n• NVL(a, b) — Oracle: returns b if a is NULL\n• ISNULL(a, b) — SQL Server: returns b if a is NULL",
    example: "SELECT name,\n  COALESCE(department, 'Unassigned') AS department,\n  COALESCE(salary, 0) AS salary,\n  NULLIF(bonus, 0) AS bonus_or_null\nFROM employees\nWHERE manager_id IS NOT NULL;",
    tags: ["null", "nulls", "is null", "is not null", "coalesce", "nullif", "ifnull", "nvl", "isnull", "null handling", "missing data", "unknown"],
  },
  {
    title: "CASE / WHEN Expression",
    content: "CASE is SQL's if/else conditional. It can be used in SELECT, WHERE, ORDER BY, and inside aggregates.\n\nTwo forms:\n\nSimple CASE:\n• CASE col WHEN val1 THEN result1 WHEN val2 THEN result2 ELSE default END\n\nSearched CASE:\n• CASE WHEN condition1 THEN result1 WHEN condition2 THEN result2 ELSE default END\n\nUse cases:\n\n• Categorize data into buckets\n• Conditional aggregation\n• Dynamic sorting\n• Pivoting rows to columns",
    example: "SELECT name, salary,\n  CASE\n    WHEN salary >= 100000 THEN 'Senior'\n    WHEN salary >= 60000 THEN 'Mid-Level'\n    ELSE 'Junior'\n  END AS level,\n  -- Conditional aggregation:\n  COUNT(CASE WHEN department = 'Engineering' THEN 1 END) AS eng_count\nFROM employees\nGROUP BY name, salary;",
    tags: ["case", "case when", "case statement", "case expression", "if else", "conditional", "switch", "when then"],
  },
  {
    title: "Stored Procedures & Functions",
    content: "A stored procedure is a precompiled set of SQL statements stored in the database.\n\nProcedure vs Function:\n\n• Procedure — called with CALL/EXEC, can modify data, doesn't need to return a value\n• Function — called in expressions, must return a value, should avoid side effects\n\nBenefits:\n\n• Performance — precompiled execution plan\n• Security — controlled data access\n• Reusability — call from multiple apps\n• Maintainability — change logic in one place",
    example: "CREATE PROCEDURE give_raise(IN dept VARCHAR(50), IN pct DECIMAL(5,2))\nBEGIN\n  UPDATE employees SET salary = salary * (1 + pct/100)\n  WHERE department = dept;\nEND;\n\nCALL give_raise('Engineering', 10.0);\n\nCREATE FUNCTION get_tax(salary DECIMAL) RETURNS DECIMAL\nBEGIN\n  RETURN salary * 0.3;\nEND;",
    tags: ["stored procedure", "stored procedures", "procedure", "function", "functions", "create procedure", "create function", "stored function", "udf", "user defined function"],
  },
  {
    title: "Database Triggers",
    content: "A trigger automatically executes when a specific event occurs on a table (INSERT, UPDATE, DELETE).\n\nTiming:\n\n• BEFORE — runs before the event (can modify/cancel)\n• AFTER — runs after the event (good for auditing)\n• INSTEAD OF — replaces the event (used with views)\n\nScope:\n\n• FOR EACH ROW — fires per affected row\n• FOR EACH STATEMENT — fires once per SQL statement\n\nUse cases: audit logging, enforcing business rules, maintaining computed columns, replication.",
    example: "CREATE TRIGGER audit_salary\nAFTER UPDATE OF salary ON employees\nFOR EACH ROW\nBEGIN\n  INSERT INTO salary_audit(employee_id, old_salary, new_salary, changed_at)\n  VALUES (OLD.id, OLD.salary, NEW.salary, NOW());\nEND;",
    tags: ["trigger", "triggers", "before trigger", "after trigger", "instead of", "audit", "database trigger", "create trigger"],
  },
  {
    title: "CREATE TABLE Statement",
    content: "CREATE TABLE defines a new table with columns, data types, and constraints.\n\nCommon data types:\n\n• INT / BIGINT — integers\n• DECIMAL(p,s) / NUMERIC — exact decimals\n• VARCHAR(n) / TEXT — variable-length strings\n• DATE / TIMESTAMP / DATETIME — date/time\n• BOOLEAN — true/false\n• JSON / JSONB — JSON data (PostgreSQL)\n\nCommon constraints:\n\n• PRIMARY KEY — unique row identifier\n• NOT NULL — cannot be empty\n• UNIQUE — no duplicates\n• DEFAULT — fallback value\n• CHECK — custom validation\n• FOREIGN KEY — references another table",
    example: "CREATE TABLE employees (\n  id SERIAL PRIMARY KEY,\n  name VARCHAR(100) NOT NULL,\n  email VARCHAR(255) UNIQUE NOT NULL,\n  department VARCHAR(50) DEFAULT 'General',\n  salary DECIMAL(10,2) CHECK (salary > 0),\n  hire_date DATE DEFAULT CURRENT_DATE,\n  manager_id INT REFERENCES employees(id)\n);",
    tags: ["create table", "create", "table", "data types", "constraints", "not null", "unique", "default", "check", "serial", "auto increment", "autoincrement", "identity"],
  },
  {
    title: "ALTER TABLE Statement",
    content: "ALTER TABLE modifies an existing table's structure.\n\nCommon operations:\n\n• ADD COLUMN — add a new column\n• DROP COLUMN — remove a column\n• RENAME COLUMN — change column name\n• ALTER COLUMN / MODIFY — change data type or constraints\n• ADD CONSTRAINT — add a new constraint\n• DROP CONSTRAINT — remove a constraint\n• RENAME TO — rename the table",
    example: "ALTER TABLE employees ADD COLUMN phone VARCHAR(20);\nALTER TABLE employees DROP COLUMN phone;\nALTER TABLE employees RENAME COLUMN name TO full_name;\nALTER TABLE employees ALTER COLUMN salary SET DEFAULT 0;\nALTER TABLE employees ADD CONSTRAINT chk_salary CHECK (salary >= 0);",
    tags: ["alter table", "alter", "modify table", "add column", "drop column", "rename column", "modify column", "change column", "alter column"],
  },
  {
    title: "INSERT, UPDATE, DELETE Statements",
    content: "DML statements for modifying data:\n\nINSERT adds new rows:\n\n• INSERT INTO table VALUES (...) — all columns\n• INSERT INTO table (col1, col2) VALUES (...) — specific columns\n• INSERT INTO ... SELECT ... — insert from another query\n\nUPDATE modifies existing rows:\n\n• Always use WHERE to avoid updating all rows!\n• Can update from a JOIN or subquery\n\nDELETE removes rows:\n\n• DELETE FROM table WHERE ... — remove matching rows\n• TRUNCATE TABLE — remove all rows (faster, no WHERE)",
    example: "INSERT INTO employees (name, department, salary)\nVALUES ('Alice', 'Engineering', 85000);\n\nINSERT INTO archive SELECT * FROM employees WHERE status = 'inactive';\n\nUPDATE employees SET salary = salary * 1.1 WHERE department = 'Sales';\n\nDELETE FROM employees WHERE hire_date < '2020-01-01';",
    tags: ["insert", "update", "delete", "insert into", "update set", "delete from", "truncate", "dml statements", "crud", "crud operations", "insert update delete", "modify data"],
  },
  {
    title: "DROP vs DELETE vs TRUNCATE",
    content: "Three ways to remove data, with different scopes:\n\nDELETE (DML):\n\n• Removes specific rows matching WHERE\n• Can be rolled back (logged operation)\n• Fires triggers\n• Slower for large datasets\n\nTRUNCATE (DDL):\n\n• Removes ALL rows from a table\n• Cannot use WHERE\n• Much faster than DELETE (minimal logging)\n• Resets auto-increment counters\n• Cannot be rolled back in most databases\n\nDROP (DDL):\n\n• Removes the entire table (structure + data)\n• Table no longer exists\n• Cannot be rolled back",
    example: "-- Remove specific rows:\nDELETE FROM employees WHERE department = 'Temp';\n\n-- Remove all rows, keep table:\nTRUNCATE TABLE logs;\n\n-- Remove entire table:\nDROP TABLE IF EXISTS temp_data;",
    tags: ["drop", "delete", "truncate", "drop vs delete", "delete vs truncate", "drop vs truncate", "drop table", "remove data", "drop delete truncate", "difference between drop delete truncate"],
  },
  {
    title: "LIKE & Pattern Matching",
    content: "LIKE matches strings against patterns in WHERE clauses.\n\nWildcards:\n\n• % — matches any number of characters (including zero)\n• _ — matches exactly one character\n\nExamples:\n\n• 'J%' — starts with J\n• '%son' — ends with son\n• '%eng%' — contains eng\n• 'J___' — J followed by exactly 3 characters\n\nAdvanced:\n\n• ILIKE — case-insensitive LIKE (PostgreSQL)\n• SIMILAR TO — regex-like patterns (PostgreSQL)\n• REGEXP / RLIKE — full regex support (MySQL)\n• ~ operator — regex matching (PostgreSQL)",
    example: "SELECT * FROM employees WHERE name LIKE 'J%';\nSELECT * FROM employees WHERE email LIKE '%@gmail.com';\nSELECT * FROM employees WHERE name LIKE '_a%';  -- 2nd char is 'a'\n\n-- Case-insensitive (PostgreSQL):\nSELECT * FROM employees WHERE name ILIKE '%john%';",
    tags: ["like", "pattern matching", "wildcard", "wildcards", "ilike", "regexp", "regex", "rlike", "similar to", "string matching", "pattern"],
  },
  {
    title: "IN, BETWEEN, EXISTS Operators",
    content: "Filtering operators for WHERE clauses:\n\nIN — checks if value matches any in a list:\n\n• WHERE col IN (val1, val2, val3)\n• WHERE col IN (SELECT ...)\n• NOT IN — excludes listed values\n\nBETWEEN — checks if value falls within a range (inclusive):\n\n• WHERE col BETWEEN low AND high\n• Works with numbers, dates, and strings\n\nEXISTS — checks if a subquery returns any rows:\n\n• WHERE EXISTS (SELECT 1 FROM ...)\n• More efficient than IN for large subqueries\n• NOT EXISTS — no matching rows",
    example: "SELECT * FROM employees\nWHERE department IN ('Sales', 'Marketing', 'Engineering');\n\nSELECT * FROM employees\nWHERE salary BETWEEN 50000 AND 100000;\n\nSELECT * FROM departments d\nWHERE EXISTS (\n  SELECT 1 FROM employees e WHERE e.dept_id = d.id AND e.salary > 90000\n);",
    tags: ["in", "between", "exists", "not in", "not exists", "in operator", "between operator", "exists operator", "in clause"],
  },
  {
    title: "SQL Data Types",
    content: "Common SQL data types across databases:\n\nNumeric:\n\n• INT / INTEGER — whole numbers (-2B to 2B)\n• BIGINT — large integers\n• SMALLINT — small integers (-32K to 32K)\n• DECIMAL(p,s) / NUMERIC — exact decimals\n• FLOAT / REAL / DOUBLE — approximate decimals\n\nString:\n\n• VARCHAR(n) — variable-length (up to n chars)\n• CHAR(n) — fixed-length (padded with spaces)\n• TEXT — unlimited length\n\nDate/Time:\n\n• DATE — date only (YYYY-MM-DD)\n• TIME — time only (HH:MM:SS)\n• TIMESTAMP / DATETIME — date + time\n• INTERVAL — time duration\n\nOther:\n\n• BOOLEAN — true/false\n• JSON / JSONB — JSON data\n• UUID — universally unique identifier\n• BLOB / BYTEA — binary data\n• ARRAY — array of values (PostgreSQL)\n• ENUM — predefined set of values",
    example: "CREATE TABLE products (\n  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,\n  name VARCHAR(200) NOT NULL,\n  price DECIMAL(10,2),\n  quantity INT DEFAULT 0,\n  metadata JSONB,\n  created_at TIMESTAMP DEFAULT NOW(),\n  is_active BOOLEAN DEFAULT true\n);",
    tags: ["data types", "data type", "int", "integer", "varchar", "char", "text", "decimal", "numeric", "float", "double", "date", "time", "timestamp", "datetime", "boolean", "json", "jsonb", "uuid", "blob", "bytea", "array", "enum", "bigint", "smallint", "serial"],
  },
  {
    title: "SQL Constraints",
    content: "Constraints enforce rules on table data to maintain integrity:\n\n• PRIMARY KEY — uniquely identifies each row (NOT NULL + UNIQUE)\n• FOREIGN KEY — references another table's primary key\n• NOT NULL — column cannot contain NULL\n• UNIQUE — all values in column must be different\n• CHECK — values must satisfy a boolean expression\n• DEFAULT — provides a fallback value if none specified\n• EXCLUDE — prevents overlapping ranges (PostgreSQL)\n\nTable-level vs column-level:\n\n• Column-level: defined inline with column\n• Table-level: defined separately, can span multiple columns",
    example: "CREATE TABLE orders (\n  id INT PRIMARY KEY,\n  customer_id INT NOT NULL REFERENCES customers(id),\n  amount DECIMAL(10,2) CHECK (amount > 0),\n  status VARCHAR(20) DEFAULT 'pending'\n    CHECK (status IN ('pending', 'shipped', 'delivered')),\n  email VARCHAR(255) UNIQUE,\n  CONSTRAINT positive_amount CHECK (amount > 0)\n);",
    tags: ["constraints", "constraint", "not null", "unique", "check", "default", "foreign key constraint", "primary key constraint", "table constraint", "column constraint", "data integrity", "integrity"],
  },
  {
    title: "String Functions",
    content: "Common string manipulation functions:\n\n• CONCAT(a, b) / || — concatenate strings\n• LENGTH(s) / LEN(s) — string length\n• UPPER(s) / LOWER(s) — change case\n• TRIM(s) / LTRIM / RTRIM — remove whitespace\n• SUBSTRING(s, start, len) / SUBSTR — extract portion\n• REPLACE(s, old, new) — replace occurrences\n• LEFT(s, n) / RIGHT(s, n) — first/last n characters\n• REVERSE(s) — reverse a string\n• POSITION(sub IN s) / CHARINDEX — find substring\n• SPLIT_PART(s, delimiter, n) — split and get nth part\n• REPEAT(s, n) — repeat string n times\n• LPAD(s, len, pad) / RPAD — pad string",
    example: "SELECT\n  UPPER(name) AS upper_name,\n  CONCAT(first_name, ' ', last_name) AS full_name,\n  LENGTH(email) AS email_length,\n  SUBSTRING(phone, 1, 3) AS area_code,\n  REPLACE(address, 'St.', 'Street') AS full_address\nFROM employees;",
    tags: ["string functions", "string", "concat", "length", "upper", "lower", "trim", "substring", "substr", "replace", "left", "right", "reverse", "position", "charindex", "split", "lpad", "rpad", "text functions"],
  },
  {
    title: "Date & Time Functions",
    content: "Common date/time functions:\n\n• NOW() / CURRENT_TIMESTAMP — current date+time\n• CURRENT_DATE / CURDATE() — current date\n• DATE_ADD / DATE_SUB / + INTERVAL — add/subtract time\n• DATEDIFF(a, b) — difference between dates\n• EXTRACT(part FROM date) / DATE_PART — get year/month/day/etc.\n• DATE_TRUNC(part, date) — truncate to precision (PostgreSQL)\n• TO_CHAR(date, format) — format as string\n• DATE_FORMAT(date, format) — format as string (MySQL)\n• AGE(a, b) — interval between dates (PostgreSQL)\n• DATEADD / DATEPART — SQL Server functions",
    example: "SELECT name, hire_date,\n  EXTRACT(YEAR FROM hire_date) AS hire_year,\n  AGE(NOW(), hire_date) AS tenure,\n  hire_date + INTERVAL '1 year' AS anniversary,\n  DATE_TRUNC('month', hire_date) AS hire_month\nFROM employees\nWHERE hire_date >= CURRENT_DATE - INTERVAL '6 months';",
    tags: ["date functions", "time functions", "date", "now", "current_timestamp", "current_date", "date_add", "date_sub", "datediff", "extract", "date_part", "date_trunc", "to_char", "date_format", "interval", "dateadd", "datetime functions"],
  },
  {
    title: "SQL Aliases",
    content: "Aliases give temporary names to tables or columns in a query. They make results more readable and simplify complex queries.\n\nColumn alias:\n\n• SELECT col AS alias_name — renames in output\n• Used in ORDER BY, but NOT in WHERE (because WHERE runs before SELECT)\n\nTable alias:\n\n• FROM table AS t — shorthand for table references\n• Essential for self-joins and correlated subqueries\n• No AS keyword needed (optional in most databases)",
    example: "SELECT\n  e.name AS employee_name,\n  d.name AS department_name,\n  e.salary * 12 AS annual_salary\nFROM employees e\nINNER JOIN departments d ON e.dept_id = d.id\nORDER BY annual_salary DESC;",
    tags: ["alias", "aliases", "as", "column alias", "table alias", "rename column", "temporary name"],
  },
  {
    title: "SQL Injection & Security",
    content: "SQL injection is an attack where malicious SQL is inserted through user input, potentially exposing or destroying data.\n\nExample of vulnerable code:\n\n• query = \"SELECT * FROM users WHERE name = '\" + input + \"'\"\n• If input = \"'; DROP TABLE users; --\" → table gets dropped!\n\nPrevention:\n\n• Parameterized queries / prepared statements (most important)\n• Input validation and sanitization\n• Least-privilege database users\n• ORM frameworks (abstract raw SQL)\n• Web application firewalls\n• Stored procedures with no dynamic SQL\n• Escaping special characters (last resort)",
    example: "-- VULNERABLE (never do this):\n-- query = \"SELECT * FROM users WHERE id = \" + user_input\n\n-- SAFE (parameterized query):\n-- Python: cursor.execute(\"SELECT * FROM users WHERE id = %s\", (user_input,))\n-- Node.js: db.query('SELECT * FROM users WHERE id = $1', [userId])\n-- Java: stmt.setInt(1, userId)\n\nPREPARE safe_query (int) AS\n  SELECT * FROM users WHERE id = $1;\nEXECUTE safe_query(42);",
    tags: ["sql injection", "injection", "security", "parameterized queries", "prepared statements", "sanitization", "xss", "attack", "vulnerability", "secure"],
  },
  {
    title: "Relational Databases vs NoSQL",
    content: "Relational (SQL) databases store data in structured tables with fixed schemas. NoSQL databases offer flexible schemas and different data models.\n\nRelational (SQL):\n\n• Tables with rows and columns\n• ACID transactions\n• Strong consistency\n• SQL query language\n• Examples: PostgreSQL, MySQL, SQL Server, Oracle, SQLite\n\nNoSQL types:\n\n• Document — JSON-like documents (MongoDB, CouchDB)\n• Key-Value — simple key→value pairs (Redis, DynamoDB)\n• Column-Family — wide-column stores (Cassandra, HBase)\n• Graph — nodes and edges (Neo4j, ArangoDB)\n\nWhen to use SQL: complex queries, transactions, data integrity, structured data\nWhen to use NoSQL: massive scale, flexible schema, unstructured data, high write throughput",
    example: "-- SQL (relational):\nSELECT u.name, COUNT(o.id) AS order_count\nFROM users u LEFT JOIN orders o ON u.id = o.user_id\nGROUP BY u.name;\n\n-- Equivalent in MongoDB (NoSQL):\n-- db.users.aggregate([\n--   { $lookup: { from: 'orders', ... } },\n--   { $group: { _id: '$name', order_count: { $sum: 1 } } }\n-- ])",
    tags: ["relational database", "rdbms", "nosql", "sql vs nosql", "mongodb", "redis", "postgresql", "mysql", "sql server", "oracle", "sqlite", "cassandra", "neo4j", "database types", "database comparison"],
  },
  {
    title: "Query Execution Order",
    content: "SQL queries are NOT executed in the order they're written. Understanding execution order is crucial for debugging and optimization.\n\nExecution order:\n\n• 1. FROM / JOIN — identify source tables\n• 2. WHERE — filter individual rows\n• 3. GROUP BY — group rows\n• 4. HAVING — filter groups\n• 5. SELECT — choose columns and compute expressions\n• 6. DISTINCT — remove duplicates\n• 7. ORDER BY — sort results\n• 8. LIMIT / OFFSET — restrict output rows\n\nThis is why:\n\n• You can't use column aliases in WHERE (SELECT hasn't run yet)\n• You CAN use aliases in ORDER BY (it runs after SELECT)\n• WHERE can't use aggregates (GROUP BY hasn't run yet)\n• HAVING can use aggregates (it runs after GROUP BY)",
    example: "-- Written order:        -- Execution order:\nSELECT department,       -- 5th\n  AVG(salary) AS avg_sal -- 5th\nFROM employees           -- 1st\nWHERE status = 'active'  -- 2nd\nGROUP BY department      -- 3rd\nHAVING AVG(salary) > 50k -- 4th\nORDER BY avg_sal DESC    -- 6th\nLIMIT 10;                -- 7th",
    tags: ["execution order", "query execution order", "order of execution", "sql order", "how sql executes", "query processing", "logical order"],
  },
  {
    title: "Common Table Expressions (CTEs)",
    content: "A CTE (WITH clause) defines a temporary named result set that exists only within a single query. It makes complex queries readable and maintainable.\n\nTypes:\n\n• Non-recursive CTE — a named subquery\n• Recursive CTE — references itself (for hierarchical/tree data)\n\nBenefits over subqueries:\n\n• More readable (named and defined upfront)\n• Can be referenced multiple times in the same query\n• Recursive CTEs can traverse hierarchies\n• Easier to debug (test CTE independently)",
    example: "WITH high_earners AS (\n  SELECT name, department, salary\n  FROM employees WHERE salary > 80000\n),\ndept_stats AS (\n  SELECT department, COUNT(*) AS cnt\n  FROM high_earners GROUP BY department\n)\nSELECT * FROM dept_stats WHERE cnt > 2;\n\n-- Recursive CTE (org chart):\nWITH RECURSIVE org_chart AS (\n  SELECT id, name, manager_id, 0 AS level\n  FROM employees WHERE manager_id IS NULL\n  UNION ALL\n  SELECT e.id, e.name, e.manager_id, oc.level + 1\n  FROM employees e JOIN org_chart oc ON e.manager_id = oc.id\n)\nSELECT * FROM org_chart;",
    tags: ["cte", "ctes", "common table expression", "with clause", "with", "recursive cte", "recursive query", "with recursive", "temporary result"],
  },
  {
    title: "EXPLAIN & Query Optimization",
    content: "EXPLAIN shows the database's execution plan for a query, helping identify performance bottlenecks.\n\nKey concepts:\n\n• Seq Scan — full table scan (slow for large tables)\n• Index Scan — uses an index (fast)\n• Nested Loop / Hash Join / Merge Join — different join strategies\n• Sort — explicit sorting operation\n• Cost — estimated resource usage\n• Rows — estimated row count\n\nOptimization tips:\n\n• Add indexes for frequently filtered/joined columns\n• Avoid SELECT * — select only needed columns\n• Use LIMIT for large result sets\n• Avoid functions on indexed columns in WHERE\n• Use EXISTS instead of IN for large subqueries\n• Consider materialized views for complex aggregations",
    example: "EXPLAIN ANALYZE\nSELECT e.name, d.name\nFROM employees e\nJOIN departments d ON e.dept_id = d.id\nWHERE e.salary > 80000;\n\n-- Output shows:\n-- Hash Join (cost=X..Y rows=Z)\n--   -> Seq Scan on employees (filter: salary > 80000)\n--   -> Hash (Index Scan on departments)",
    tags: ["explain", "explain analyze", "query plan", "execution plan", "query optimization", "optimization", "performance", "slow query", "seq scan", "index scan", "query tuning"],
  },
  {
    title: "Database Schema Design",
    content: "Schema design is the process of defining tables, columns, relationships, and constraints for a database.\n\nBest practices:\n\n• Use meaningful names (snake_case for columns/tables)\n• Each table should represent one entity\n• Use appropriate data types (smallest sufficient type)\n• Always have a primary key\n• Define foreign keys for relationships\n• Add indexes for query patterns\n• Normalize to 3NF, then selectively denormalize for performance\n\nRelationship types:\n\n• One-to-One — user → profile (rare)\n• One-to-Many — department → employees (most common)\n• Many-to-Many — students ↔ courses (requires junction table)",
    example: "-- One-to-Many:\nCREATE TABLE departments (id SERIAL PRIMARY KEY, name VARCHAR(50));\nCREATE TABLE employees (id SERIAL PRIMARY KEY, dept_id INT REFERENCES departments(id));\n\n-- Many-to-Many:\nCREATE TABLE students (id SERIAL PRIMARY KEY, name VARCHAR(100));\nCREATE TABLE courses (id SERIAL PRIMARY KEY, title VARCHAR(100));\nCREATE TABLE enrollments (\n  student_id INT REFERENCES students(id),\n  course_id INT REFERENCES courses(id),\n  PRIMARY KEY (student_id, course_id)\n);",
    tags: ["schema design", "schema", "database design", "er diagram", "erd", "entity relationship", "one to many", "many to many", "one to one", "junction table", "relationship", "relationships", "table design"],
  },
];

export function findKnowledgeEntry(query: string): KnowledgeEntry | null {
  const lower = query.toLowerCase().replace(/[?!.,;:'"]/g, "").trim();
  const normalized = lower.replace(/\//g, " ").replace(/\s+/g, " ");

  for (const entry of SQL_KNOWLEDGE) {
    const titleLower = entry.title.toLowerCase();
    if (normalized === titleLower || lower.includes(titleLower)) return entry;
  }

  for (const entry of SQL_KNOWLEDGE) {
    for (const tag of entry.tags) {
      if (normalized === tag || normalized.includes(tag)) return entry;
      if (tag.includes(normalized) && normalized.length > 3) return entry;
    }
  }

  const words = normalized.split(/\s+/).filter(w => w.length > 2);
  const stopWords = new Set(["what", "how", "why", "the", "does", "can", "you", "explain", "tell", "about", "mean", "meaning", "define", "definition", "difference", "between", "use", "used", "work", "works", "please", "help", "understand"]);
  const meaningful = words.filter(w => !stopWords.has(w));

  if (meaningful.length > 0) {
    let bestMatch: KnowledgeEntry | null = null;
    let bestScore = 0;

    for (const entry of SQL_KNOWLEDGE) {
      let score = 0;
      const allText = [entry.title.toLowerCase(), ...entry.tags].join(" ");

      for (const word of meaningful) {
        if (allText.includes(word)) score += 1;
        for (const tag of entry.tags) {
          if (tag === word) score += 3;
          if (tag.split(/\s+/).includes(word)) score += 2;
        }
      }

      if (score > bestScore) {
        bestScore = score;
        bestMatch = entry;
      }
    }

    if (bestScore >= 2) return bestMatch;
  }

  return null;
}

export function classifyIntent(query: string): "sql_generation" | "explanation" {
  const trimmed = query.trim();
  const lower = trimmed.toLowerCase();

  const explanationStarters = [
    "what is", "what are", "what's", "whats",
    "explain", "describe", "define", "tell me about",
    "how does", "how do", "how to", "why is", "why do", "why are",
    "difference between", "compare", "meaning of",
    "can you explain", "help me understand",
    "what does", "when to use", "when should",
    "types of", "list of", "examples of",
  ];

  for (const starter of explanationStarters) {
    if (lower.startsWith(starter) || lower.includes(starter)) return "explanation";
  }

  if (findKnowledgeEntry(trimmed)) return "explanation";

  const sqlGenerationPatterns = [
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

  for (const pattern of sqlGenerationPatterns) {
    if (pattern.test(trimmed)) return "sql_generation";
  }

  if (lower.endsWith("?")) return "explanation";

  return "sql_generation";
}

export function getAllTopicNames(): string[] {
  return SQL_KNOWLEDGE.map(e => e.title);
}
