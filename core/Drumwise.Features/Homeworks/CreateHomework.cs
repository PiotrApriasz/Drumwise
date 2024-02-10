using Drumwise.Application.Common.Errors;
using Drumwise.Application.Common.Models;
using Drumwise.Application.Entities;
using Drumwise.Features.Homeworks.Events;
using Drumwise.Infrastructure.Data;
using MediatR;

namespace Drumwise.Features.Homeworks;

public record CreateHomeworkCommand(string AssignedTo,
                                    string Title,
                                    string Exercise,
                                    DateTime Deadline) : IRequest<(Result, Guid)>;

public class CreateHomeworkHandler(ApplicationDbContext context) : IRequestHandler<CreateHomeworkCommand, (Result, Guid)>
{
    public async Task<(Result, Guid)> Handle(CreateHomeworkCommand request, CancellationToken cancellationToken)
    {
        var entity = new Homework()
        {
            AssignedTo = request.AssignedTo,
            HomeworkTitle = request.Title,
            Exercise = request.Exercise,
            Deadline = request.Deadline
        };
        
        entity.AddDomainEvent(new HomeworkCreatedEvent(entity));

        context.Homeworks.Add(entity);

        await context.SaveChangesAsync(cancellationToken).ConfigureAwait(false);

        return (Result.Success(ResultType.Ok), entity.Id);
    }
}
                                    