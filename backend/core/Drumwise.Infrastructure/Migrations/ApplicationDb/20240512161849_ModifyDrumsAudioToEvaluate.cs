using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace Drumwise.Infrastructure.Migrations.ApplicationDb
{
    /// <inheritdoc />
    public partial class ModifyDrumsAudioToEvaluate : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<string>(
                name: "GoogleDriveFileId",
                table: "DrumsAudiosToEvaluate",
                type: "TEXT",
                nullable: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "GoogleDriveFileId",
                table: "DrumsAudiosToEvaluate");
        }
    }
}
