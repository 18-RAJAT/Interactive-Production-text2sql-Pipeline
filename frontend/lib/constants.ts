export const SAMPLE_SCHEMAS = [
  {
    label: "Employees",
    value: `CREATE TABLE employees (
  id INTEGER PRIMARY KEY,
  name TEXT,
  salary REAL,
  department TEXT,
  hire_date TEXT
);`,
  },
  {
    label: "Products & Orders",
    value: `CREATE TABLE products (
  id INTEGER PRIMARY KEY,
  name TEXT,
  price REAL,
  category TEXT
);

CREATE TABLE orders (
  id INTEGER PRIMARY KEY,
  product_id INTEGER,
  customer_name TEXT,
  quantity INTEGER,
  order_date TEXT
);`,
  },
  {
    label: "Students & Courses",
    value: `CREATE TABLE students (
  id INTEGER PRIMARY KEY,
  name TEXT,
  age INTEGER,
  major TEXT,
  gpa REAL
);

CREATE TABLE enrollments (
  student_id INTEGER,
  course_name TEXT,
  grade TEXT,
  semester TEXT
);`,
  },
  {
    label: "WikiSQL Style",
    value: `CREATE TABLE "table1" (
  "state_territory" TEXT,
  "format" TEXT,
  "current_slogan" TEXT,
  "current_series" TEXT,
  "notes" TEXT
);`,
  },
];

export const STORAGE_KEY = "text-to-sql-history";