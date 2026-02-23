set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

count=0

find "$PROJECT_ROOT" \
  -type f \
  \( -name "*.py" -o -name "*.ts" -o -name "*.tsx" -o -name "*.yaml" -o -name "*.yml" \
     -o -name "*.json" -o -name "*.css" -o -name "*.js" -o -name "*.md" -o -name "*.txt" \) \
  ! -path "*/node_modules/*" \
  ! -path "*/.next/*" \
  ! -path "*/__pycache__/*" \
  ! -path "*/venv/*" \
  ! -path "*/.git/*" \
  -print0 | while IFS= read -r -d '' file; do

  [ ! -s "$file" ] && continue

  trimmed=$(perl -e '
    local $/;
    my $content = <STDIN>;
    $content =~ s/\n+\z/\n/;
    print $content;
  ' < "$file")

  if [ "$(wc -c < "$file")" -ne "$(printf "%s" "$trimmed" | wc -c)" ]; then
    printf "%s" "$trimmed" > "$file"
    echo "trimmed: ${file#$PROJECT_ROOT/}"
    count=$((count + 1))
  fi

done

echo "done. $count file(s) trimmed."