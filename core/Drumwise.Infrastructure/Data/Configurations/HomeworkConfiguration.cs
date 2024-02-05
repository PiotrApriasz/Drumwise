using Drumwise.Application.Entities;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Drumwise.Infrastructure.Data.Configurations;

public class HomeworkConfiguration : IEntityTypeConfiguration<Homework>
{
    public void Configure(EntityTypeBuilder<Homework> builder)
    {
        builder.Property(p => p.HomeworkTitle)
            .IsRequired();
        
        builder.Property(p => p.Deadline)
            .IsRequired();
        
        builder.Property(p => p.Exercise)
            .IsRequired();
        
        builder.Property(p => p.AssignedTo)
            .IsRequired();
    }
}