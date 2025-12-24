# Setup dependencies
setup:
    uv sync

# Ingest books into vector DB
ingest:
    uv run python3 -m book_writer.ingest

# Generate outlines for the next two books
outlines:
    uv run python3 -m book_writer.outline

# Generate "The Winds of Winter"
# Copies the specific outline to outline.txt, runs the generator, and moves output
write-winds:
    @echo "Preparing to write The Winds of Winter..."
    cp winds_of_winter_outline.txt outline.txt
    uv run python3 -m book_writer.write
    rm -rf winds_of_winter_generated
    rm outline.txt
    mv generated_book winds_of_winter_generated
    @echo "Done! Book saved to winds_of_winter_generated/"

# Generate "A Dream of Spring"
# First, ingests the newly written "Winds of Winter" into the DB for context
write-dream:
    @echo "Ingesting 'The Winds of Winter' into knowledge base..."
    -uv run python3 -m book_writer.ingest --dir winds_of_winter_generated
    @echo "Preparing to write A Dream of Spring..."
    cp dream_of_spring_outline.txt outline.txt
    uv run python3 -m book_writer.write
    rm -rf dream_of_spring_generated
    rm outline.txt
    mv generated_book dream_of_spring_generated
    @echo "Done! Book saved to dream_of_spring_generated/"

review-winds:
    uv run python3 -m book_writer.review --dir winds_of_winter_generated
    @echo "Review saved to winds_of_winter_generated_review.md"

review-dream:
    uv run python3 -m book_writer.review --dir dream_of_spring_generated
    @echo "Review saved to dream_of_spring_generated_review.md"

improve-winds:
    uv run python3 -m book_writer.improve --book_dir winds_of_winter_generated --review_file reviews/winds_of_winter_generated_review.md

improve-dream:
    uv run python3 -m book_writer.improve --book_dir dream_of_spring_generated --review_file reviews/dream_of_spring_generated_review.md

summarize-winds:
    uv run python3 -m book_writer.summarize --book_dir winds_of_winter_generated_improved

summarize-dream:
    uv run python3 -m book_writer.summarize --book_dir dream_of_spring_generated_improved
